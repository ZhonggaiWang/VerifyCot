"""Binary pre-commit alignment verification for VoCoT coordinates."""

from typing import Any, Callable, Dict, List, Mapping, Optional

from utils.coordinate_intervention import box_iou

from ..contracts import (
    ALIGNMENT_OUTPUT_SCHEMA,
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
    GrounderBackend,
    VerifierFailClosedError,
    VerificationRequest,
)
from .alignment_policy import AlignmentRoutingDecision, AlignmentRoutingPolicy
from .coordinate_rollout import CoordinateRolloutBase, PrecommitInferenceResult
from .expert_dispatch import (
    ExpertDispatcher,
    ExpertNotConfiguredError,
    ExpertUnavailableError,
)


ALIGNMENT_EVENT_SCHEMA = 'vocot_precommit_alignment_event_v1'


class PrecommitGroundingController(CoordinateRolloutBase):
    """Verify every candidate with the binary alignment contract.

    Accepted candidates are committed unchanged.  Rejected and uncertain
    candidates are sent to the configured Grounder according to
    :class:`AlignmentRoutingPolicy`.  Four-way action routing intentionally
    lives in :mod:`grounding_control.four_way`.
    """

    def __init__(
            self,
            model,
            tokenizer,
            batch_factory: Callable[[], Dict[str, Any]],
            verifier: AlignmentVerifierBackend,
            grounder: Optional[GrounderBackend],
            sample_id: str,
            alignment_routing_policy: AlignmentRoutingPolicy,
            log_path: Optional[str] = None,
            sample_context: Optional[Mapping[str, Any]] = None,
            expert_dispatcher: Optional[ExpertDispatcher] = None,
            missing_expert_policy: str = 'fail_open'):
        if verifier is None:
            raise ValueError('verifier backend is required')
        if alignment_routing_policy is None:
            raise ValueError('alignment_routing_policy is required')
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
        self.alignment_routing_policy = alignment_routing_policy
        self.expert_dispatcher = expert_dispatcher or ExpertDispatcher(
            grounder=grounder,
        )
        self.missing_expert_policy = missing_expert_policy

    def _verify_alignment(
            self,
            request: VerificationRequest) -> AlignmentVerifierOutput:
        """Run the sanitized binary verifier boundary."""

        verify_alignment = getattr(self.verifier, 'verify_alignment', None)
        if not callable(verify_alignment):
            raise TypeError(
                'binary routing requires verifier.verify_alignment()'
            )
        try:
            output = verify_alignment(request.alignment_request())
        except VerifierFailClosedError:
            # This is an explicit backend policy boundary, not an ordinary
            # unavailable-verifier result.  Propagate it so fail-closed cannot
            # be silently converted back into the controller's default
            # fail-open behavior.
            raise
        except Exception as error:
            return AlignmentVerifierOutput.unknown(
                error=f'{type(error).__name__}: {error}',
                score_semantics='unavailable_backend_exception',
                metadata={
                    'backend_exception': True,
                    'backend_exception_type': type(error).__name__,
                    'backend_class': type(self.verifier).__name__,
                },
            )
        if not isinstance(output, AlignmentVerifierOutput):
            raise TypeError(
                'verify_alignment() must return AlignmentVerifierOutput'
            )
        return output

    def _alignment_routing_decision(
            self,
            output: AlignmentVerifierOutput) -> AlignmentRoutingDecision:
        return self.alignment_routing_policy.decide(output)

    def run(
            self,
            max_new_tokens: int = 1024,
            temperature: float = 0.0) -> PrecommitInferenceResult:
        """Run a full CoT and verify every generated coordinate."""

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
            alignment_output = self._verify_alignment(request)
            metadata = dict(alignment_output.metadata)
            alignment_decision = self._alignment_routing_decision(
                alignment_output
            )

            committed_box = request.candidate_bbox
            committed_tokens = candidate_tokens
            grounder_invoked = False
            grounder_attempted = False
            grounder_result = None
            expert_result = None
            missing_expert_error = None
            missing_expert_metadata = None
            router_action = alignment_decision.routing_reason

            if alignment_decision.requires_grounder:
                try:
                    grounder_attempted = (
                        not hasattr(self.expert_dispatcher, 'grounder')
                        or self.expert_dispatcher.grounder is not None
                    )
                    expert_result = self.expert_dispatcher.dispatch_grounder(
                        request.grounding_request(),
                        action='relocate',
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
                        f'{alignment_decision.routing_reason}'
                        '_unavailable_accept'
                    )

            if expert_result is not None:
                committed_tokens, committed_box = (
                    self._encode_expert_coordinate(expert_result.bbox)
                )
                grounder_invoked = True
                grounder_result = expert_result
                router_action = expert_result.metadata.get(
                    'router_action',
                    alignment_decision.routing_reason,
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
                'box_refiner_invoked': False,
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
                'committed_coordinate_text': self.tokenizer.decode(
                    committed_tokens,
                    skip_special_tokens=False,
                ),
                'committed_box': self._box_list(committed_box),
                'committed_iou_to_gt': committed_iou_to_gt,
                'committed_feature_will_be_injected_on_clean_replay': True,
                'event_schema': ALIGNMENT_EVENT_SCHEMA,
                'verifier_output_schema': ALIGNMENT_OUTPUT_SCHEMA,
                'alignment_score': alignment_decision.alignment_score,
                'alignment_score_kind': alignment_decision.score_kind,
                'score_semantics': alignment_decision.score_semantics,
                'raw_alignment_score': alignment_output.alignment_score,
                'raw_alignment_score_kind': alignment_output.score_kind,
                'raw_score_semantics': alignment_output.score_semantics,
                'reject_threshold': alignment_decision.reject_threshold,
                'accept_threshold': alignment_decision.accept_threshold,
                'decision_band': alignment_decision.band,
                'system_action': alignment_decision.system_action,
                'routing_reason': alignment_decision.routing_reason,
                'verifier_abstained': alignment_output.abstained,
                'verifier_failure': (
                    alignment_decision.band == 'verifier_failure'
                ),
                'verifier_error': alignment_output.error,
                'verifier_metadata': metadata,
                'routing_policy_metadata': dict(alignment_decision.metadata),
                'grounder_requested': alignment_decision.requires_grounder,
                'grounder_attempted': grounder_attempted,
                'grounder_succeeded': grounder_invoked,
                'candidate_committed': expert_result is None,
            }
            events.append(event)
            self._write_event(event)

            # The next iteration replays exactly the selected coordinate; its
            # normal REFbind feature then becomes part of persistent history.
            persistent = h_t_ids + committed_tokens


__all__ = [
    'ALIGNMENT_EVENT_SCHEMA',
    'PrecommitGroundingController',
    'PrecommitInferenceResult',
]
