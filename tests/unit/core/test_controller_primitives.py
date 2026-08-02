"""CPU-only tests for the pre-commit verifier primitives."""

import json
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from grounding_control.core.coordinate_rollout import (
    _ForcePrefixProcessor,
    GenerationBoundary as _GenerationBoundary,
    _StopAfterNewCoordinate,
)
from grounding_control.legacy.repair_controller import _ForcePrefixSuppressNextBocProcessor
from grounding_control.core.precommit_controller import (
    PrecommitGroundingController,
)
from grounding_control.contracts import (
    AlignmentVerifierOutput,
    GroundingResult,
    VerificationRequest,
    VerifierFailClosedError,
)
from grounding_control.core import AlignmentRoutingPolicy
from grounding_control.core import (
    ExpertDispatchResult as ExpertRouteResult,
    ExpertDispatcher,
    ExpertUnavailableError,
)
from grounding_control.four_way import (
    ActionVerifierOutput,
    FourWayPrecommitGroundingController,
    RoutingPolicy,
)
from grounding_control.legacy import (
    StoredOracleVerifier,
    VerificationLookup,
    VerificationResult,
    build_repair_prompt,
)
from grounding_control.verifiers import RemoteAlignmentVerifierBackend


class _CoordinateTokenizer:
    """Tiny reversible tokenizer for coordinate-controller CPU tests."""

    boc_token_id = 1000
    eoc_token_id = 1001
    eos_token_id = 1002
    character_offset = 2000

    def convert_tokens_to_ids(self, token):
        return {
            DEFAULT_BOC_TOKEN: self.boc_token_id,
            DEFAULT_EOC_TOKEN: self.eoc_token_id,
        }.get(token)

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        token_ids = []
        offset = 0
        while offset < len(text):
            if text.startswith(DEFAULT_BOC_TOKEN, offset):
                token_ids.append(self.boc_token_id)
                offset += len(DEFAULT_BOC_TOKEN)
            elif text.startswith(DEFAULT_EOC_TOKEN, offset):
                token_ids.append(self.eoc_token_id)
                offset += len(DEFAULT_EOC_TOKEN)
            else:
                token_ids.append(self.character_offset + ord(text[offset]))
                offset += 1
        return SimpleNamespace(input_ids=token_ids)

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        pieces = []
        for token_id in token_ids:
            if token_id == self.boc_token_id:
                pieces.append(DEFAULT_BOC_TOKEN)
            elif token_id == self.eoc_token_id:
                pieces.append(DEFAULT_EOC_TOKEN)
            elif token_id == self.eos_token_id:
                pieces.append('</s>')
            else:
                pieces.append(chr(token_id - self.character_offset))
        return ''.join(pieces)


class VerifierPrimitiveTests(unittest.TestCase):
    def _oracle_file(self, records):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / 'oracle.jsonl'
        path.write_text('\n'.join(json.dumps(record) for record in records) + '\n', encoding='utf-8')
        self.addCleanup(directory.cleanup)
        return path

    def test_result_validation_rejects_illegal_accept(self):
        with self.assertRaises(ValueError):
            VerificationResult('aligned', 'wrong_object', 1.0)
        with self.assertRaises(ValueError):
            VerificationResult('aligned', 'none', 1.1)
        self.assertEqual(
            VerificationResult('misaligned', 'unsupported', 0.9).reason,
            'unsupported',
        )
        with self.assertRaises(ValueError):
            VerificationResult('uncertain', 'unsupported', 0.9)
        self.assertEqual(
            VerificationResult('misaligned', 'ambiguous', 0.9).reason,
            'ambiguous',
        )
        self.assertEqual(VerificationResult.unknown().verdict, 'unknown')

    def test_oracle_lookup_and_missing_record_never_accept(self):
        path = self._oracle_file([{
            'sample_id': 's1', 'grounding_step': 2, 'attempt_index': 0,
            'candidate_bbox': [0.1, 0.2, 0.3, 0.4],
            'verifier_output': {'verdict': 'aligned', 'reason': 'none', 'confidence': 1.0},
        }])
        verifier = StoredOracleVerifier(str(path))
        matched = verifier.verify('s1', 2, 0, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(matched.result.verdict, 'aligned')
        self.assertFalse(matched.missing_oracle_record)
        missing = verifier.verify('s1', 2, 1, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(missing.result.verdict, 'uncertain')
        self.assertTrue(missing.missing_oracle_record)

    def test_candidate_mismatch_is_uncertain_not_reused_oracle_answer(self):
        path = self._oracle_file([{
            'sample_id': 's1', 'grounding_step': 1, 'attempt_index': 0,
            'candidate_bbox': [0.1, 0.2, 0.3, 0.4],
            'verifier_output': {'verdict': 'aligned', 'reason': 'none', 'confidence': 1.0},
        }])
        decision = StoredOracleVerifier(str(path)).verify('s1', 1, 0, [0.5, 0.2, 0.7, 0.4])
        self.assertEqual(decision.result.verdict, 'uncertain')
        self.assertTrue(decision.oracle_candidate_mismatch)

    def test_feedback_prompts_preserve_reference_and_stop_at_opening_tag(self):
        reference = 'the bar for series B in 2019'
        box = [0.1, 0.2, 0.3, 0.4]
        for mode, reason in [('binary_feedback', 'wrong_object'), ('typed_feedback', 'wrong_object'),
                             ('concise_typed_feedback', 'wrong_object'),
                             ('separated_reference_feedback', 'wrong_object'),
                             ('typed_feedback', 'partial_coverage'), ('typed_feedback', 'ambiguous')]:
            prompt = build_repair_prompt(mode, reference, box, reason)
            self.assertIn(reference, prompt)
            self.assertIn(f'{DEFAULT_BOC_TOKEN}0.100,0.200,0.300,0.400{DEFAULT_EOC_TOKEN}', prompt)
            self.assertTrue(prompt.endswith(DEFAULT_BOC_TOKEN))
        self.assertEqual(build_repair_prompt('blind_retry', reference, box, 'wrong_object'), '')

    def test_prefix_replay_and_coordinate_stop_boundary(self):
        processor = _ForcePrefixProcessor(prompt_length=3, prefix_ids=[7, 8])
        scores = torch.zeros((1, 10))
        first = processor(torch.tensor([[1, 2, 3]]), scores.clone())
        self.assertEqual(first.argmax(dim=-1).item(), 7)
        second = processor(torch.tensor([[1, 2, 3, 7]]), scores.clone())
        self.assertEqual(second.argmax(dim=-1).item(), 8)
        released = processor(torch.tensor([[1, 2, 3, 7, 8]]), scores.clone())
        self.assertTrue(processor.released)
        self.assertTrue(torch.equal(released, scores))

        stopper = _StopAfterNewCoordinate(minimum_sequence_length=5, eoc_token_id=9)
        self.assertFalse(stopper(torch.tensor([[1, 2, 3, 7, 9]]), scores))
        self.assertTrue(stopper(torch.tensor([[1, 2, 3, 7, 8, 9]]), scores))

    def test_expert_coordinate_is_encoded_and_round_trip_validated_locally(self):
        tokenizer = _CoordinateTokenizer()
        controller = PrecommitGroundingController.__new__(
            PrecommitGroundingController
        )
        controller.tokenizer = tokenizer
        controller.boc_token_id = tokenizer.boc_token_id
        controller.eoc_token_id = tokenizer.eoc_token_id

        token_ids, encoded_box = controller._encode_expert_coordinate(
            [0.1234, 0.2344, 0.6784, 0.7894]
        )

        self.assertEqual(
            tokenizer.decode(token_ids),
            '<coor>0.123,0.234,0.678,0.789</coor>',
        )
        self.assertEqual(encoded_box, (0.123, 0.234, 0.678, 0.789))

    def test_routed_expert_box_enters_only_the_next_clean_replay(self):
        tokenizer = _CoordinateTokenizer()
        h_t_ids = tokenizer('Find the cup ').input_ids
        candidate_tokens = tokenizer(
            '<coor>0.700,0.700,0.900,0.900</coor>'
        ).input_ids
        expected_expert_tokens = tokenizer(
            '<coor>0.200,0.300,0.600,0.700</coor>'
        ).input_ids
        replayed_prefixes = []

        class ModelThatMustNotRunForExpertEncoding:
            def condition_completion(self, *args, **kwargs):
                raise AssertionError(
                    'expert coordinate encoding must not call Volcano'
                )

        class RelocateVerifier:
            def verify_action(self, request):
                del request
                return ActionVerifierOutput(
                    predicted_action='relocate',
                    action_probabilities=None,
                    confidence=1.0,
                )

        class OracleExpertDispatcher:
            def dispatch(self, decision, request, verification):
                del decision, request, verification
                return ExpertRouteResult(
                    bbox=(0.2, 0.3, 0.6, 0.7),
                    source='oracle_gt_box',
                    confidence=1.0,
                    expert_role='grounder',
                    action='relocate',
                    metadata={
                        'router_action': 'routed_to_oracle_grounder',
                        'evaluation_reference_box': [0.2, 0.3, 0.6, 0.7],
                    },
                )

        controller = FourWayPrecommitGroundingController.__new__(
            FourWayPrecommitGroundingController
        )
        controller.model = ModelThatMustNotRunForExpertEncoding()
        controller.tokenizer = tokenizer
        controller.boc_token_id = tokenizer.boc_token_id
        controller.eoc_token_id = tokenizer.eoc_token_id
        controller.verifier = RelocateVerifier()
        controller.routing_policy = RoutingPolicy(confidence_threshold=0.0)
        controller.expert_dispatcher = OracleExpertDispatcher()
        controller.missing_expert_policy = 'error'
        controller.sample_id = 'sample'
        controller.sample_context = {}
        controller.log_path = None

        def fake_generate(
                self, persistent_ids, max_new_tokens, temperature):
            del self, max_new_tokens, temperature
            persistent = list(persistent_ids)
            replayed_prefixes.append(persistent)
            if len(replayed_prefixes) == 1:
                generated = h_t_ids + candidate_tokens
                return _GenerationBoundary(
                    generated_ids=generated,
                    candidate_span=(
                        len(h_t_ids),
                        len(generated) - 1,
                    ),
                    candidate_box=(0.7, 0.7, 0.9, 0.9),
                    replayed_bound_box_count=0,
                )
            return _GenerationBoundary(
                generated_ids=persistent + [tokenizer.eos_token_id],
                candidate_span=None,
                candidate_box=None,
                replayed_bound_box_count=1,
            )

        controller._generate_until_next_coordinate = MethodType(
            fake_generate,
            controller,
        )
        result = controller.run(max_new_tokens=128, temperature=0.0)

        self.assertEqual(len(replayed_prefixes), 2)
        self.assertEqual(
            replayed_prefixes[1],
            h_t_ids + expected_expert_tokens,
        )
        self.assertEqual(
            result.events[0]['committed_box'],
            [0.2, 0.3, 0.6, 0.7],
        )
        self.assertEqual(
            result.events[0]['expert_coordinate_commit_mode'],
            'local_roundtrip_then_single_clean_replay',
        )
        self.assertFalse(
            result.events[0]['expert_coordinate_extra_model_forward']
        )
        self.assertTrue(
            result.events[0][
                'committed_feature_will_be_injected_on_clean_replay'
            ]
        )

    def test_four_way_terminal_abstain_keeps_candidate_uncommitted(self):
        tokenizer = _CoordinateTokenizer()
        h_t_ids = tokenizer('Find the cup ').input_ids
        candidate_tokens = tokenizer(
            '<coor>0.700,0.700,0.900,0.900</coor>'
        ).input_ids
        generation_calls = []

        class AbstainingVerifier:
            def verify_action(self, request):
                del request
                return ActionVerifierOutput.unknown(
                    error='ambiguous candidate',
                )

        controller = FourWayPrecommitGroundingController.__new__(
            FourWayPrecommitGroundingController
        )
        controller.model = SimpleNamespace()
        controller.tokenizer = tokenizer
        controller.boc_token_id = tokenizer.boc_token_id
        controller.eoc_token_id = tokenizer.eoc_token_id
        controller.verifier = AbstainingVerifier()
        controller.routing_policy = RoutingPolicy(
            confidence_threshold=0.8,
            unknown_action='abstain',
        )
        controller.expert_dispatcher = SimpleNamespace()
        controller.missing_expert_policy = 'error'
        controller.sample_id = 'sample'
        controller.sample_context = {}
        controller.log_path = None

        def fake_generate(
                self, persistent_ids, max_new_tokens, temperature):
            del self, max_new_tokens, temperature
            generation_calls.append(list(persistent_ids))
            generated = h_t_ids + candidate_tokens
            return _GenerationBoundary(
                generated_ids=generated,
                candidate_span=(len(h_t_ids), len(generated) - 1),
                candidate_box=(0.7, 0.7, 0.9, 0.9),
                replayed_bound_box_count=0,
            )

        controller._generate_until_next_coordinate = MethodType(
            fake_generate,
            controller,
        )
        result = controller.run(max_new_tokens=128, temperature=0.0)

        self.assertEqual(generation_calls, [[]])
        self.assertEqual(result.status, 'routing_abstained')
        self.assertEqual(result.generated_ids, h_t_ids)
        self.assertEqual(result.response, 'Find the cup ')
        self.assertEqual(len(result.events), 1)
        event = result.events[0]
        self.assertEqual(event['candidate_box'], [0.7, 0.7, 0.9, 0.9])
        self.assertFalse(event['coordinate_committed'])
        self.assertTrue(event['terminal_uncommitted'])
        self.assertFalse(event['candidate_committed'])
        self.assertIsNone(event['committed_coordinate_text'])
        self.assertIsNone(event['committed_box'])
        self.assertFalse(
            event['committed_feature_will_be_injected_on_clean_replay']
        )

    def test_uncertain_alignment_band_calls_grounder_before_refbind(self):
        tokenizer = _CoordinateTokenizer()
        h_t_ids = tokenizer('Find the cup ').input_ids
        candidate_tokens = tokenizer(
            '<coor>0.700,0.700,0.900,0.900</coor>'
        ).input_ids
        replacement_tokens = tokenizer(
            '<coor>0.200,0.300,0.600,0.700</coor>'
        ).input_ids
        replayed_prefixes = []

        class UncertainVerifier:
            def verify_alignment(self, request):
                del request
                return AlignmentVerifierOutput(
                    alignment_score=0.5,
                    score_semantics='test_probability',
                    score_kind='calibrated_probability',
                    metadata={'candidate_iou_to_gt': 0.1},
                )

        class Grounder:
            calls = 0

            def ground(self, request):
                self.calls += 1
                self.last_request = request
                return GroundingResult(
                    bbox=(0.2, 0.3, 0.6, 0.7),
                    source='test_grounder',
                )

        grounder = Grounder()
        controller = PrecommitGroundingController.__new__(
            PrecommitGroundingController
        )
        controller.model = SimpleNamespace()
        controller.tokenizer = tokenizer
        controller.boc_token_id = tokenizer.boc_token_id
        controller.eoc_token_id = tokenizer.eoc_token_id
        controller.verifier = UncertainVerifier()
        controller.alignment_routing_policy = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
        )
        controller.expert_dispatcher = ExpertDispatcher(grounder=grounder)
        controller.missing_expert_policy = 'error'
        controller.sample_id = 'sample'
        controller.sample_context = {}
        controller.log_path = None

        def fake_generate(
                self, persistent_ids, max_new_tokens, temperature):
            del self, max_new_tokens, temperature
            persistent = list(persistent_ids)
            replayed_prefixes.append(persistent)
            if len(replayed_prefixes) == 1:
                generated = h_t_ids + candidate_tokens
                return _GenerationBoundary(
                    generated_ids=generated,
                    candidate_span=(len(h_t_ids), len(generated) - 1),
                    candidate_box=(0.7, 0.7, 0.9, 0.9),
                    replayed_bound_box_count=0,
                )
            return _GenerationBoundary(
                generated_ids=persistent + [tokenizer.eos_token_id],
                candidate_span=None,
                candidate_box=None,
                replayed_bound_box_count=1,
            )

        controller._generate_until_next_coordinate = MethodType(
            fake_generate,
            controller,
        )
        result = controller.run(max_new_tokens=128, temperature=0.0)

        self.assertEqual(grounder.calls, 1)
        self.assertEqual(replayed_prefixes[1], h_t_ids + replacement_tokens)
        event = result.events[0]
        self.assertEqual(event['decision_band'], 'uncertain')
        self.assertEqual(event['system_action'], 'call_grounder')
        self.assertEqual(
            event['routing_reason'],
            'uncertainty_grounder_fallback',
        )
        self.assertTrue(event['grounder_requested'])
        self.assertTrue(event['grounder_attempted'])
        self.assertTrue(event['grounder_invoked'])
        self.assertTrue(event['grounder_succeeded'])
        self.assertFalse(event['candidate_committed'])
        self.assertIsNone(event['committed_iou_to_gt'])
        self.assertEqual(event['verifier_output_schema'], 'vocot_alignment_score_v1')
        self.assertEqual(event['alignment_score'], 0.5)
        self.assertEqual(
            event['alignment_score_kind'],
            'calibrated_probability',
        )
        self.assertEqual(event['raw_alignment_score'], 0.5)
        self.assertEqual(
            event['raw_alignment_score_kind'],
            'calibrated_probability',
        )
        self.assertNotIn('predicted_action', event)

    def test_alignment_verifier_failure_fails_open_without_grounder(self):
        tokenizer = _CoordinateTokenizer()
        h_t_ids = tokenizer('Find the cup ').input_ids
        candidate_tokens = tokenizer(
            '<coor>0.200,0.300,0.600,0.700</coor>'
        ).input_ids
        replayed_prefixes = []

        class FailedVerifier:
            def verify_alignment(self, request):
                del request
                raise RuntimeError('worker timeout')

        controller = PrecommitGroundingController.__new__(
            PrecommitGroundingController
        )
        controller.model = SimpleNamespace()
        controller.tokenizer = tokenizer
        controller.boc_token_id = tokenizer.boc_token_id
        controller.eoc_token_id = tokenizer.eoc_token_id
        controller.verifier = FailedVerifier()
        controller.alignment_routing_policy = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
        )
        controller.expert_dispatcher = ExpertDispatcher()
        controller.missing_expert_policy = 'error'
        controller.sample_id = 'sample'
        controller.sample_context = {}
        controller.log_path = None

        def fake_generate(
                self, persistent_ids, max_new_tokens, temperature):
            del self, max_new_tokens, temperature
            persistent = list(persistent_ids)
            replayed_prefixes.append(persistent)
            if len(replayed_prefixes) == 1:
                generated = h_t_ids + candidate_tokens
                return _GenerationBoundary(
                    generated_ids=generated,
                    candidate_span=(len(h_t_ids), len(generated) - 1),
                    candidate_box=(0.2, 0.3, 0.6, 0.7),
                    replayed_bound_box_count=0,
                )
            return _GenerationBoundary(
                generated_ids=persistent + [tokenizer.eos_token_id],
                candidate_span=None,
                candidate_box=None,
                replayed_bound_box_count=1,
            )

        controller._generate_until_next_coordinate = MethodType(
            fake_generate,
            controller,
        )
        result = controller.run(max_new_tokens=128, temperature=0.0)

        self.assertEqual(replayed_prefixes[1], h_t_ids + candidate_tokens)
        event = result.events[0]
        self.assertEqual(event['decision_band'], 'verifier_failure')
        self.assertEqual(event['system_action'], 'accept_candidate')
        self.assertTrue(event['verifier_failure'])
        self.assertFalse(event['grounder_requested'])
        self.assertFalse(event['grounder_attempted'])
        self.assertFalse(event['grounder_invoked'])
        self.assertFalse(event['grounder_succeeded'])
        self.assertTrue(event['candidate_committed'])
        self.assertIn('RuntimeError: worker timeout', event['verifier_error'])
        self.assertTrue(event['verifier_metadata']['backend_exception'])

    def test_remote_fail_closed_error_escapes_binary_controller(self):
        class FailedClient:
            def request(self, payload, timeout=None):
                del payload, timeout
                raise RuntimeError('worker timeout')

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'image.png'
            image_path.write_bytes(b'placeholder')
            request = VerificationRequest(
                sample_id='sample',
                grounding_step=1,
                object_reference='the cup',
                candidate_bbox=(0.2, 0.3, 0.6, 0.7),
                candidate_coordinate_text=(
                    '<coor>0.200,0.300,0.600,0.700</coor>'
                ),
                generated_ids=(1, 2),
                candidate_span=(0, 1),
                sample_context={'image_path': str(image_path)},
            )
            controller = PrecommitGroundingController.__new__(
                PrecommitGroundingController
            )
            controller.verifier = RemoteAlignmentVerifierBackend(
                FailedClient(),
                fail_open=False,
            )

            with self.assertRaises(VerifierFailClosedError) as raised:
                controller._verify_alignment(request)

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertIn('worker timeout', str(raised.exception))

    def test_rejected_alignment_grounder_failure_fails_open_and_is_counted(self):
        tokenizer = _CoordinateTokenizer()
        h_t_ids = tokenizer('Find the cup ').input_ids
        candidate_tokens = tokenizer(
            '<coor>0.200,0.300,0.600,0.700</coor>'
        ).input_ids
        replayed_prefixes = []

        class RejectVerifier:
            def verify_alignment(self, request):
                del request
                return AlignmentVerifierOutput(
                    alignment_score=0.1,
                    score_semantics='test_probability',
                    score_kind='calibrated_probability',
                )

        class UnavailableGrounder:
            def ground(self, request):
                del request
                raise ExpertUnavailableError('no object found')

        controller = PrecommitGroundingController.__new__(
            PrecommitGroundingController
        )
        controller.model = SimpleNamespace()
        controller.tokenizer = tokenizer
        controller.boc_token_id = tokenizer.boc_token_id
        controller.eoc_token_id = tokenizer.eoc_token_id
        controller.verifier = RejectVerifier()
        controller.alignment_routing_policy = AlignmentRoutingPolicy(
            reject_threshold=0.2,
            accept_threshold=0.8,
        )
        controller.expert_dispatcher = ExpertDispatcher(
            grounder=UnavailableGrounder()
        )
        controller.missing_expert_policy = 'fail_open'
        controller.sample_id = 'sample'
        controller.sample_context = {}
        controller.log_path = None

        def fake_generate(
                self, persistent_ids, max_new_tokens, temperature):
            del self, max_new_tokens, temperature
            persistent = list(persistent_ids)
            replayed_prefixes.append(persistent)
            if len(replayed_prefixes) == 1:
                generated = h_t_ids + candidate_tokens
                return _GenerationBoundary(
                    generated_ids=generated,
                    candidate_span=(len(h_t_ids), len(generated) - 1),
                    candidate_box=(0.2, 0.3, 0.6, 0.7),
                    replayed_bound_box_count=0,
                )
            return _GenerationBoundary(
                generated_ids=persistent + [tokenizer.eos_token_id],
                candidate_span=None,
                candidate_box=None,
                replayed_bound_box_count=1,
            )

        controller._generate_until_next_coordinate = MethodType(
            fake_generate,
            controller,
        )
        result = controller.run(max_new_tokens=128, temperature=0.0)

        self.assertEqual(replayed_prefixes[1], h_t_ids + candidate_tokens)
        event = result.events[0]
        self.assertEqual(event['decision_band'], 'reject')
        self.assertTrue(event['grounder_requested'])
        self.assertTrue(event['grounder_attempted'])
        self.assertFalse(event['grounder_succeeded'])
        self.assertFalse(event['grounder_invoked'])
        self.assertTrue(event['candidate_committed'])
        self.assertEqual(event['missing_expert_error'], 'no object found')

    def test_failure_continuation_masks_only_first_free_boc(self):
        processor = _ForcePrefixSuppressNextBocProcessor(
            prompt_length=3, prefix_ids=[7], boc_token_id=9
        )
        scores = torch.zeros((1, 10))
        forced = processor(torch.tensor([[1, 2, 3]]), scores.clone())
        self.assertEqual(forced.argmax(dim=-1).item(), 7)
        first_free = processor(torch.tensor([[1, 2, 3, 7]]), scores.clone())
        self.assertTrue(processor.suppressed_first_free_boc)
        self.assertTrue(torch.isneginf(first_free[0, 9]))
        later_free = processor(torch.tensor([[1, 2, 3, 7, 4]]), scores.clone())
        self.assertFalse(torch.isneginf(later_free[0, 9]))

    def test_routing_policy_routes_only_confident_misalignment(self):
        controller = FourWayPrecommitGroundingController.__new__(
            FourWayPrecommitGroundingController
        )
        controller.verifier_confidence_threshold = 0.8
        self.assertFalse(controller._should_route(VerificationLookup(
            VerificationResult('aligned', 'none', 1.0)
        )))
        self.assertFalse(controller._should_route(VerificationLookup(
            VerificationResult('misaligned', 'wrong_object', 0.79)
        )))
        self.assertFalse(controller._should_route(VerificationLookup(
            VerificationResult.uncertain()
        )))
        # Unsupported is no longer a learned action and fails open by default.
        self.assertFalse(controller._should_route(VerificationLookup(
            VerificationResult.unsupported(1.0)
        )))
        self.assertTrue(controller._should_route(VerificationLookup(
            VerificationResult('misaligned', 'wrong_object', 0.8)
        )))


if __name__ == '__main__':
    unittest.main()
