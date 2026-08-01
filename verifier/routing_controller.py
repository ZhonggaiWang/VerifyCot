"""Online verifier--expert routing for VoCoT coordinate generation.

The controller has one commit rule: a generated coordinate is stopped at its
closing ``</coor>`` before its REFbind feature enters the next model forward.
The verifier then either accepts that candidate or routes it to the
specialized Grounder/BoxRefiner selected by policy. Only the selected
coordinate is cleanly replayed and therefore
bound into the persistent VoCoT trajectory.

This module intentionally contains no prompt-based self-repair, retry loop, or
sandbox REFbind ablation.  Those historical experiments live under
``verifier.legacy``.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from utils.coordinate_intervention import box_iou, find_coordinate_spans

from .contracts import (
    ACTION_OUTPUT_SCHEMA,
    ActionVerifierBackend,
    ActionVerifierOutput,
    BoxRefinerBackend,
    GrounderBackend,
    VerificationRequest,
    VerifierBackend,
    validate_normalized_box,
)
from .adapters import legacy_lookup_to_action_output
from .expert_router import (
    ExpertNotConfiguredError,
    ExpertUnavailableError,
    ExpertRouter,
)
from .routing_policy import RoutingDecision, RoutingPolicy
from .types import Box, VerificationLookup


class _ForcePrefixProcessor(LogitsProcessor):
    """Deterministically replay a committed generated-token prefix."""

    def __init__(self, prompt_length: int, prefix_ids: Sequence[int]):
        self.prompt_length = int(prompt_length)
        self.prefix_ids = list(prefix_ids)
        self.released = not self.prefix_ids

    @staticmethod
    def _force(scores: torch.FloatTensor, token_id: int) -> torch.FloatTensor:
        scores.fill_(float('-inf'))
        scores[:, int(token_id)] = 0
        return scores

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[0, self.prompt_length:].tolist()
        if len(generated) < len(self.prefix_ids):
            if generated != self.prefix_ids[:len(generated)]:
                raise RuntimeError('committed prefix diverged during clean replay')
            return self._force(scores, self.prefix_ids[len(generated)])
        if generated[:len(self.prefix_ids)] != self.prefix_ids:
            raise RuntimeError('clean replay did not preserve the committed prefix')
        self.released = True
        return scores


class _StopAfterNewCoordinate(StoppingCriteria):
    """Stop after a newly generated EOC and before its REFbind forward pass."""

    def __init__(self, minimum_sequence_length: int, eoc_token_id: int):
        self.minimum_sequence_length = int(minimum_sequence_length)
        self.eoc_token_id = int(eoc_token_id)
        self.triggered = False

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs) -> bool:
        self.triggered = bool(
            input_ids.shape[1] > self.minimum_sequence_length
            and input_ids[0, -1].item() == self.eoc_token_id
        )
        return self.triggered


class _StopAtForcedCoordinateEnd(StoppingCriteria):
    """Stop exactly after a grounding backend's forced closing EOC."""

    def __init__(self, target_sequence_length: int, eoc_token_id: int):
        self.target_sequence_length = int(target_sequence_length)
        self.eoc_token_id = int(eoc_token_id)
        self.triggered = False

    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs) -> bool:
        self.triggered = bool(
            input_ids.shape[1] == self.target_sequence_length
            and input_ids[0, -1].item() == self.eoc_token_id
        )
        return self.triggered


@dataclass
class _GenerationBoundary:
    generated_ids: List[int]
    candidate_span: Optional[Tuple[int, int]]
    candidate_box: Optional[Box]
    replayed_bound_box_count: int


@dataclass
class RoutingInferenceResult:
    response: str
    generated_ids: List[int]
    status: str
    events: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'response': self.response,
            'generated_ids': self.generated_ids,
            'status': self.status,
            'events': self.events,
        }


class RoutingController:
    """Verify every coordinate and route rejection types to specialists."""

    def __init__(
            self,
            model,
            tokenizer,
            batch_factory: Callable[[], Dict[str, Any]],
            verifier: Union[VerifierBackend, ActionVerifierBackend],
            grounder: Optional[GrounderBackend],
            sample_id: str,
            verifier_confidence_threshold: float = 0.8,
            log_path: Optional[str] = None,
            sample_context: Optional[Mapping[str, Any]] = None,
            box_refiner: Optional[BoxRefinerBackend] = None,
            routing_policy: Optional[RoutingPolicy] = None,
            expert_router: Optional[ExpertRouter] = None,
            missing_expert_policy: str = 'fail_open'):
        if not sample_id:
            raise ValueError('sample_id is required')
        if verifier is None:
            raise ValueError('verifier backend is required')
        if not 0.0 <= float(verifier_confidence_threshold) <= 1.0:
            raise ValueError('verifier_confidence_threshold must be in [0, 1]')
        if missing_expert_policy not in {'fail_open', 'error'}:
            raise ValueError('missing_expert_policy must be fail_open or error')

        self.model = model
        self.tokenizer = tokenizer
        self.batch_factory = batch_factory
        self.verifier = verifier
        self.grounder = grounder
        self.box_refiner = box_refiner
        self.sample_id = str(sample_id)
        self.verifier_confidence_threshold = float(verifier_confidence_threshold)
        self.routing_policy = routing_policy or RoutingPolicy(
            confidence_threshold=self.verifier_confidence_threshold,
            unsupported_action='no_action',
        )
        self.expert_router = expert_router or ExpertRouter(
            grounder=grounder,
            box_refiner=box_refiner,
        )
        self.missing_expert_policy = missing_expert_policy
        self.log_path = Path(log_path) if log_path else None
        self.sample_context = dict(sample_context or {})
        self.boc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_BOC_TOKEN)
        self.eoc_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_EOC_TOKEN)
        if self.boc_token_id is None or self.eoc_token_id is None:
            raise ValueError('tokenizer does not provide VoCoT coordinate tokens')

    @staticmethod
    def _validate_box(values: Sequence[float]) -> Box:
        return validate_normalized_box(values)

    def _box_from_span(
            self,
            generated_ids: Sequence[int],
            span: Tuple[int, int]) -> Box:
        text = self.tokenizer.decode(
            generated_ids[span[0]:span[1] + 1],
            skip_special_tokens=False,
        )
        match = re.search(
            re.escape(DEFAULT_BOC_TOKEN)
            + r'\s*([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*,\s*'
            + r'([01](?:\.\d+)?)\s*,\s*([01](?:\.\d+)?)\s*'
            + re.escape(DEFAULT_EOC_TOKEN),
            text,
        )
        if match is None:
            raise ValueError(f'could not parse completed generated coordinate: {text!r}')
        return self._validate_box(tuple(float(value) for value in match.groups()))

    def _object_reference(self, h_t_ids: Sequence[int]) -> str:
        """Extract the local text since the previous completed coordinate."""

        last_eoc = -1
        for index, token_id in enumerate(h_t_ids):
            if token_id == self.eoc_token_id:
                last_eoc = index
        text = self.tokenizer.decode(
            h_t_ids[last_eoc + 1:],
            skip_special_tokens=False,
        )
        text = re.sub(r'\s+', ' ', text).strip()
        return text[-400:] or 'the current object reference'

    def _generate_until_next_coordinate(
            self,
            persistent_ids: Sequence[int],
            max_new_tokens: int,
            temperature: float) -> _GenerationBoundary:
        """Replay committed history and stop at the next uncommitted EOC."""

        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        persistent = list(persistent_ids)
        processor = _ForcePrefixProcessor(prompt_length, persistent)
        stopper = _StopAfterNewCoordinate(
            minimum_sequence_length=prompt_length + len(persistent),
            eoc_token_id=self.eoc_token_id,
        )
        _, _, sequences = self.model.condition_completion(
            batch,
            avoid_image_gen=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            logits_processor=LogitsProcessorList([processor]),
            stopping_criteria=StoppingCriteriaList([stopper]),
            record_bound_boxes=True,
        )
        generated = sequences[0, prompt_length:].detach().cpu().tolist()
        if generated[:len(persistent)] != persistent:
            raise RuntimeError('generation diverged from committed history')

        spans = find_coordinate_spans(
            generated,
            self.boc_token_id,
            self.eoc_token_id,
        )
        candidate_span = None
        if stopper.triggered:
            candidates = [span for span in spans if span[0] >= len(persistent)]
            if not candidates:
                raise RuntimeError('stopped at </coor> without a new coordinate span')
            candidate_span = candidates[-1]
        candidate_box = (
            None
            if candidate_span is None
            else self._box_from_span(generated, candidate_span)
        )

        expected_bound_count = len(find_coordinate_spans(
            persistent,
            self.boc_token_id,
            self.eoc_token_id,
        ))
        bound_box_count = len(getattr(self.model, 'last_bound_boxes', None) or [])
        if bound_box_count != expected_bound_count:
            raise RuntimeError(
                'candidate coordinate entered REFbind before routing: '
                f'expected {expected_bound_count} committed boxes, '
                f'observed {bound_box_count}'
            )
        return _GenerationBoundary(
            generated_ids=generated,
            candidate_span=candidate_span,
            candidate_box=candidate_box,
            replayed_bound_box_count=bound_box_count,
        )

    def _force_expert_box(
            self,
            h_t_ids: Sequence[int],
            box: Sequence[float],
            max_new_tokens: int,
            temperature: float) -> _GenerationBoundary:
        """Force an expert result as coordinate text without pre-committing it."""

        normalized_box = self._validate_box(box)
        payload = ','.join(f'{value:.3f}' for value in normalized_box)
        suffix_ids = [self.boc_token_id] + self.tokenizer(
            payload + DEFAULT_EOC_TOKEN,
            add_special_tokens=False,
        ).input_ids
        if suffix_ids[-1] != self.eoc_token_id:
            raise RuntimeError('could not tokenize expert coordinate closing tag')

        batch = self.batch_factory()
        prompt_length = int(batch['input_ids'].shape[-1])
        forced_prefix = list(h_t_ids) + suffix_ids
        processor = _ForcePrefixProcessor(prompt_length, forced_prefix)
        stopper = _StopAtForcedCoordinateEnd(
            target_sequence_length=prompt_length + len(forced_prefix),
            eoc_token_id=self.eoc_token_id,
        )
        _, _, sequences = self.model.condition_completion(
            batch,
            avoid_image_gen=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            logits_processor=LogitsProcessorList([processor]),
            stopping_criteria=StoppingCriteriaList([stopper]),
            record_bound_boxes=True,
        )
        generated = sequences[0, prompt_length:].detach().cpu().tolist()
        if not stopper.triggered or generated != forced_prefix:
            raise RuntimeError('failed to force the expert coordinate')

        spans = find_coordinate_spans(
            generated,
            self.boc_token_id,
            self.eoc_token_id,
        )
        candidates = [span for span in spans if span[0] >= len(h_t_ids)]
        if not candidates:
            raise RuntimeError('forced expert result has no coordinate span')
        candidate_span = candidates[-1]
        candidate_box = self._box_from_span(generated, candidate_span)
        rounded_requested_box = tuple(
            float(f'{value:.3f}') for value in normalized_box
        )
        if any(
                abs(actual - expected) > 1e-6
                for actual, expected in zip(candidate_box, rounded_requested_box)):
            raise RuntimeError(
                'forced coordinate text differs from the expert box'
            )

        expected_bound_count = len(find_coordinate_spans(
            h_t_ids,
            self.boc_token_id,
            self.eoc_token_id,
        ))
        bound_box_count = len(getattr(self.model, 'last_bound_boxes', None) or [])
        if bound_box_count != expected_bound_count:
            raise RuntimeError('expert coordinate entered REFbind before commit')
        return _GenerationBoundary(
            generated_ids=generated,
            candidate_span=candidate_span,
            candidate_box=candidate_box,
            replayed_bound_box_count=bound_box_count,
        )

    def _should_route(
            self,
            output: Union[
                ActionVerifierOutput,
                VerificationLookup,
            ]) -> bool:
        """Compatibility predicate backed by the semantic routing policy."""

        return self._routing_decision(output).requires_expert

    def _routing_decision(
            self,
            output: Union[
                ActionVerifierOutput,
                VerificationLookup,
            ]) -> RoutingDecision:
        policy = getattr(self, 'routing_policy', None)
        if policy is None:
            # Supports old tests and code constructing the controller via
            # ``__new__`` and setting only the historical threshold field.
            policy = RoutingPolicy(
                confidence_threshold=self.verifier_confidence_threshold,
                unsupported_action='no_action',
            )
        return policy.decide(output)

    def _verify_action(
            self,
            request: VerificationRequest) -> ActionVerifierOutput:
        """Normalize new and historical backends at one explicit boundary."""

        verify_action = getattr(self.verifier, 'verify_action', None)
        if callable(verify_action):
            output = verify_action(request)
            if not isinstance(output, ActionVerifierOutput):
                raise TypeError(
                    'verify_action() must return ActionVerifierOutput'
                )
            return output

        verify = getattr(self.verifier, 'verify', None)
        if not callable(verify):
            raise TypeError(
                'verifier must implement verify_action() or legacy verify()'
            )
        lookup = verify(request)
        if not isinstance(lookup, VerificationLookup):
            raise TypeError('legacy verify() must return VerificationLookup')
        unsupported_action = getattr(
            self.routing_policy,
            'unsupported_action',
            'no_action',
        )
        return legacy_lookup_to_action_output(
            lookup,
            unsupported_action=unsupported_action,
        )

    @staticmethod
    def _box_list(box: Optional[Box]):
        return None if box is None else [float(value) for value in box]

    def _write_event(self, event: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')

    def run(
            self,
            max_new_tokens: int = 1024,
            temperature: float = 0.0) -> RoutingInferenceResult:
        """Run a full CoT while verifying every generated coordinate."""

        if max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        persistent: List[int] = []
        events: List[Dict[str, Any]] = []
        grounding_step = 0

        while True:
            boundary = self._generate_until_next_coordinate(
                persistent_ids=persistent,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if boundary.candidate_span is None:
                return RoutingInferenceResult(
                    response=self.tokenizer.decode(
                        boundary.generated_ids,
                        skip_special_tokens=False,
                    ),
                    generated_ids=boundary.generated_ids,
                    status='ok',
                    events=events,
                )
            if boundary.candidate_box is None:
                raise RuntimeError('coordinate boundary has no parseable candidate box')

            grounding_step += 1
            candidate_span = boundary.candidate_span
            h_t_ids = boundary.generated_ids[:candidate_span[0]]
            candidate_tokens = boundary.generated_ids[
                candidate_span[0]:candidate_span[1] + 1
            ]
            candidate_coordinate_text = self.tokenizer.decode(
                candidate_tokens,
                skip_special_tokens=False,
            )
            request = VerificationRequest(
                sample_id=self.sample_id,
                grounding_step=grounding_step,
                object_reference=self._object_reference(h_t_ids),
                candidate_bbox=boundary.candidate_box,
                candidate_coordinate_text=candidate_coordinate_text,
                generated_ids=tuple(boundary.generated_ids),
                candidate_span=candidate_span,
                sample_context=self.sample_context,
            )
            verifier_output = self._verify_action(request) #verifier接受Requset，输出 correct / misalign...
            metadata = dict(verifier_output.metadata)
            routing_decision = self._routing_decision(verifier_output)
            #决定到底路由结果是什么


            committed_box = request.candidate_bbox
            committed_tokens = candidate_tokens
            grounder_invoked = False
            box_refiner_invoked = False
            grounder_result = None
            expert_result = None
            missing_expert_error = None
            router_action = routing_decision.router_action
            if routing_decision.action == 'no_action':
                router_action = metadata.get(
                    'accept_router_action',
                    routing_decision.router_action,
                )

            if routing_decision.action == 'abstain':
                event = {
                    'sample_id': self.sample_id,
                    'grounding_step': grounding_step,
                    'h_t_ends_before_coor': True,
                    'object_reference': request.object_reference,
                    'candidate_coordinate_text': candidate_coordinate_text,
                    'candidate_box': self._box_list(request.candidate_bbox),
                    'candidate_refbind_uncommitted': True,
                    'predicted_action': verifier_output.predicted_action,
                    'verifier_output_schema': ACTION_OUTPUT_SCHEMA,
                    'action_probabilities': (
                        None
                        if verifier_output.action_probabilities is None
                        else dict(verifier_output.action_probabilities)
                    ),
                    'verifier_abstained': verifier_output.abstained,
                    'policy_abstained': (
                        routing_decision.verifier_abstained
                    ),
                    'verdict': routing_decision.verifier_verdict,
                    'reason': routing_decision.verifier_reason,
                    'confidence': float(verifier_output.confidence),
                    'verifier_error': verifier_output.error,
                    'verifier_metadata': metadata,
                    'routing_decision': routing_decision.action,
                    'router_action': routing_decision.router_action,
                    'expert_role': None,
                    'grounder_invoked': False,
                    'box_refiner_invoked': False,
                    'candidate_committed': False,
                }
                events.append(event)
                self._write_event(event)
                return RoutingInferenceResult(
                    response=self.tokenizer.decode(
                        h_t_ids,
                        skip_special_tokens=False,
                    ),
                    generated_ids=h_t_ids,
                    status='routing_abstained',
                    events=events,
                )

            if routing_decision.requires_expert:
                try:
                    expert_result = self.expert_router.route(
                        routing_decision,
                        request,
                        verifier_output,
                    )
                except (
                    ExpertNotConfiguredError,
                    ExpertUnavailableError,
                ) as error:
                    if self.missing_expert_policy == 'error':
                        raise
                    missing_expert_error = str(error)
                    router_action = (
                        f'{routing_decision.router_action}_unavailable_accept'
                    )

            if expert_result is not None:
                forced = self._force_expert_box(
                    h_t_ids=h_t_ids,
                    box=expert_result.bbox,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if forced.candidate_span is None or forced.candidate_box is None:
                    raise RuntimeError('expert result did not produce a coordinate')
                committed_box = forced.candidate_box
                committed_tokens = forced.generated_ids[
                    forced.candidate_span[0]:forced.candidate_span[1] + 1
                ]
                grounder_invoked = expert_result.expert_role == 'grounder'
                box_refiner_invoked = (
                    expert_result.expert_role == 'box_refiner'
                )
                grounder_result = expert_result if grounder_invoked else None
                router_action = expert_result.metadata.get(
                    'router_action',
                    routing_decision.router_action,
                )

            committed_iou_to_gt = metadata.get('candidate_iou_to_gt')
            if expert_result is not None:
                evaluation_reference_box = expert_result.metadata.get(
                    'evaluation_reference_box'
                )
                if evaluation_reference_box is not None:
                    committed_iou_to_gt = box_iou(
                        committed_box,
                        self._validate_box(evaluation_reference_box),
                    )

            expert_metadata = (
                {}
                if expert_result is None
                else dict(expert_result.metadata)
            )
            resolved_match = expert_metadata.get(
                'oracle_resolution_matched'
            )
            event = {
                'sample_id': self.sample_id,
                'grounding_step': grounding_step,
                'h_t_ends_before_coor': True,
                'object_reference': request.object_reference,
                'candidate_coordinate_text': candidate_coordinate_text,
                'candidate_box': self._box_list(request.candidate_bbox),
                'candidate_refbind_uncommitted': True,
                'predicted_action': verifier_output.predicted_action,
                'verifier_output_schema': ACTION_OUTPUT_SCHEMA,
                'action_probabilities': (
                    None
                    if verifier_output.action_probabilities is None
                    else dict(verifier_output.action_probabilities)
                ),
                'verifier_abstained': verifier_output.abstained,
                'policy_abstained': routing_decision.verifier_abstained,
                'verdict': routing_decision.verifier_verdict,
                'reason': routing_decision.verifier_reason,
                'confidence': float(verifier_output.confidence),
                'verifier_error': verifier_output.error,
                'verifier_metadata': metadata,
                # Compatibility fields retained for existing oracle summaries.
                'match_status': (
                    metadata.get('match_status')
                    or (
                        'matched_unique_explicit_target'
                        if resolved_match else None
                    )
                ),
                'match_reason': (
                    metadata.get('match_reason')
                    or expert_metadata.get('oracle_resolution_reason')
                ),
                'match_context': (
                    metadata.get('match_context')
                    or expert_metadata.get('oracle_resolution_context')
                ),
                'target_object': (
                    metadata.get('target_object')
                    or expert_metadata.get('target_object')
                ),
                'matched_alias': (
                    metadata.get('matched_alias')
                    or expert_metadata.get('matched_alias')
                ),
                'oracle_target_box': (
                    metadata.get('oracle_target_box')
                    or expert_metadata.get('oracle_target_box')
                ),
                'candidate_iou_to_gt': (
                    metadata.get('candidate_iou_to_gt')
                    if metadata.get('candidate_iou_to_gt') is not None
                    else expert_metadata.get('candidate_iou_to_gt')
                ),
                'iou_threshold': metadata.get('iou_threshold'),
                'routing_decision': routing_decision.action,
                'routing_policy_metadata': dict(routing_decision.metadata),
                'router_action': router_action,
                'expert_role': (
                    None
                    if expert_result is None
                    else expert_result.expert_role
                ),
                'expert_source': (
                    None if expert_result is None else expert_result.source
                ),
                'expert_confidence': (
                    None
                    if expert_result is None
                    else float(expert_result.confidence)
                ),
                'expert_metadata': (
                    None
                    if expert_result is None
                    else expert_metadata
                ),
                'missing_expert_error': missing_expert_error,
                'grounder_invoked': grounder_invoked,
                'box_refiner_invoked': box_refiner_invoked,
                'grounder_source': (
                    None if grounder_result is None else grounder_result.source
                ),
                'grounder_confidence': (
                    None if grounder_result is None
                    else float(grounder_result.confidence)
                ),
                'grounder_metadata': (
                    None if grounder_result is None
                    else dict(grounder_result.metadata)
                ),
                'committed_coordinate_text': self.tokenizer.decode(
                    committed_tokens,
                    skip_special_tokens=False,
                ),
                'committed_box': self._box_list(committed_box),
                'committed_iou_to_gt': committed_iou_to_gt,
                'committed_feature_will_be_injected_on_clean_replay': True,
            }
            events.append(event)
            self._write_event(event)

            # The following iteration cleanly replays this selected coordinate.
            # Its native REFbind feature is then part of the persistent history.
            persistent = h_t_ids + committed_tokens
