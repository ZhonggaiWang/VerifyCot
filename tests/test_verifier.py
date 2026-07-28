"""CPU-only tests for the pre-commit verifier primitives."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from constants import DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN
from verifier.controller import (
    _ForcePrefixProcessor,
    _ForcePrefixSuppressNextBocProcessor,
    _StopAfterNewCoordinate,
)
from verifier.prompts import (
    build_repair_prompt,
    build_repair_prompt_text_only_q,
)
from verifier.stored_oracle import StoredOracleVerifier
from verifier.types import VerificationResult


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

    def test_text_only_feedback_has_no_completed_coordinate_span(self):
        for mode in (
            'binary_feedback', 'typed_feedback', 'concise_typed_feedback',
            'separated_reference_feedback', 'separated_reference_feedback_v2',
        ):
            prompt = build_repair_prompt_text_only_q(
                'the target object', [0.1, 0.2, 0.3, 0.4], 'wrong_object', mode
            )
            self.assertIn('Rejected coordinate (text only): [0.100,0.200,0.300,0.400]', prompt)
            self.assertNotIn(DEFAULT_EOC_TOKEN, prompt)
            self.assertTrue(prompt.endswith(DEFAULT_BOC_TOKEN))

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


if __name__ == '__main__':
    unittest.main()
