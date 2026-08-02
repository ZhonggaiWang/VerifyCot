"""Archived four-action pre-commit routing controller."""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from utils.coordinate_intervention import box_iou

from ..contracts import GrounderBackend, VerificationRequest
from ..core.coordinate_rollout import (
    CoordinateRolloutBase,
    PrecommitInferenceResult,
)
from ..core.expert_dispatch import (
    ExpertNotConfiguredError,
    ExpertUnavailableError,
)
from ..legacy.contracts import VerifierBackend
from ..legacy.verdicts import VerificationLookup
from .adapters import legacy_lookup_to_action_output
from .contracts import (
    ACTION_OUTPUT_SCHEMA,
    ActionVerifierBackend,
    ActionVerifierOutput,
    BoxRefinerBackend,
)
from .expert_dispatch import FourWayExpertDispatcher
from .routing_policy import RoutingDecision, RoutingPolicy


class FourWayPrecommitGroundingController(CoordinateRolloutBase):
    """Route four verifier actions to a Grounder or BoxRefiner."""

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
            expert_dispatcher: Optional[FourWayExpertDispatcher] = None,
            missing_expert_policy: str = 'fail_open'):
        if verifier is None:
            raise ValueError('verifier backend is required')
        if not 0.0 <= float(verifier_confidence_threshold) <= 1.0:
            raise ValueError('verifier_confidence_threshold must be in [0, 1]')
        if missing_expert_policy not in {'fail_open', 'error'}:
            raise ValueError('missing_expert_policy must be fail_open or error')
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            batch_factory=batch_factory,
            sample_id=sample_id,
            log_path=log_path,
            sample_context=sample_context,
        )
        self.verifier = verifier
        self.grounder = grounder
        self.box_refiner = box_refiner
        self.verifier_confidence_threshold = float(
            verifier_confidence_threshold
        )
        self.routing_policy = routing_policy or RoutingPolicy(
            confidence_threshold=self.verifier_confidence_threshold,
            unsupported_action='no_action',
        )
        self.expert_dispatcher = (
            expert_dispatcher
            or FourWayExpertDispatcher(
                grounder=grounder,
                box_refiner=box_refiner,
            )
        )
        self.missing_expert_policy = missing_expert_policy

    def _should_route(
            self,
            output: Union[
                ActionVerifierOutput,
                VerificationLookup,
            ]) -> bool:
        return self._routing_decision(output).requires_expert

    def _routing_decision(
            self,
            output: Union[
                ActionVerifierOutput,
                VerificationLookup,
            ]) -> RoutingDecision:
        policy = getattr(self, 'routing_policy', None)
        if policy is None:
            policy = RoutingPolicy(
                confidence_threshold=self.verifier_confidence_threshold,
                unsupported_action='no_action',
            )
        return policy.decide(output)

    def _verify_action(
            self,
            request: VerificationRequest) -> ActionVerifierOutput:
        """Normalize action and historical verdict backends at one boundary."""

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
        return legacy_lookup_to_action_output(
            lookup,
            unsupported_action=self.routing_policy.unsupported_action,
        )

    def run(
            self,
            max_new_tokens: int = 1024,
            temperature: float = 0.0) -> PrecommitInferenceResult:
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
                return PrecommitInferenceResult(
                    response=self.tokenizer.decode(
                        boundary.generated_ids,
                        skip_special_tokens=False,
                    ),
                    generated_ids=boundary.generated_ids,
                    status='ok',
                    events=events,
                )
            if boundary.candidate_box is None:
                raise RuntimeError(
                    'coordinate boundary has no parseable candidate box'
                )

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
            verifier_output = self._verify_action(request)
            metadata = dict(verifier_output.metadata)
            routing_decision = self._routing_decision(verifier_output)

            committed_box = request.candidate_bbox
            committed_tokens = candidate_tokens
            grounder_invoked = False
            box_refiner_invoked = False
            grounder_result = None
            expert_result = None
            missing_expert_error = None
            missing_expert_metadata = None
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
                    'policy_abstained': routing_decision.verifier_abstained,
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
                    # This is an auditable terminal decision, not a generated
                    # coordinate in the returned trajectory.  ``routing_infer``
                    # keeps it in ``events`` while excluding it from the strict
                    # text/committed-box/REFbind equality loop.
                    'coordinate_committed': False,
                    'terminal_uncommitted': True,
                    'committed_coordinate_text': None,
                    'committed_box': None,
                    'committed_feature_will_be_injected_on_clean_replay': False,
                }
                events.append(event)
                self._write_event(event)
                return PrecommitInferenceResult(
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
                    expert_result = self.expert_dispatcher.dispatch(
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
                    missing_expert_metadata = dict(
                        getattr(error, 'metadata', {}) or {}
                    )
                    router_action = (
                        f'{routing_decision.router_action}_unavailable_accept'
                    )

            if expert_result is not None:
                committed_tokens, committed_box = (
                    self._encode_expert_coordinate(expert_result.bbox)
                )
                grounder_invoked = expert_result.expert_role == 'grounder'
                box_refiner_invoked = (
                    expert_result.expert_role == 'box_refiner'
                )
                grounder_result = expert_result if grounder_invoked else None
                router_action = expert_result.metadata.get(
                    'router_action',
                    routing_decision.router_action,
                )

            committed_iou_to_gt = (
                metadata.get('candidate_iou_to_gt')
                if expert_result is None
                else None
            )
            if expert_result is not None:
                evaluation_reference_box = (
                    expert_result.metadata.get('evaluation_reference_box')
                    or metadata.get('oracle_target_box')
                )
                if evaluation_reference_box is not None:
                    committed_iou_to_gt = box_iou(
                        committed_box,
                        self._validate_box(evaluation_reference_box),
                    )

            expert_metadata = (
                {} if expert_result is None else dict(expert_result.metadata)
            )
            resolved_match = expert_metadata.get('oracle_resolution_matched')
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
                    None if expert_result is None else expert_result.expert_role
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
                    None if expert_result is None else expert_metadata
                ),
                'expert_coordinate_commit_mode': (
                    None
                    if expert_result is None
                    else 'local_roundtrip_then_single_clean_replay'
                ),
                'expert_coordinate_extra_model_forward': False,
                'missing_expert_error': missing_expert_error,
                'missing_expert_metadata': missing_expert_metadata,
                'grounder_invoked': grounder_invoked,
                'box_refiner_invoked': box_refiner_invoked,
                'grounder_source': (
                    None if grounder_result is None else grounder_result.source
                ),
                'grounder_confidence': (
                    None
                    if grounder_result is None
                    else float(grounder_result.confidence)
                ),
                'grounder_metadata': (
                    None
                    if grounder_result is None
                    else dict(grounder_result.metadata)
                ),
                'candidate_committed': expert_result is None,
                'coordinate_committed': True,
                'terminal_uncommitted': False,
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
            persistent = h_t_ids + committed_tokens


__all__ = ['FourWayPrecommitGroundingController']
