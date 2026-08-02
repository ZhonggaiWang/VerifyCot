"""Ensure the formal Qwen binary path has no four-way action dependency."""

import unittest

from grounding_control.verifiers.qwen25_vl import prompt as binary_prompt
from grounding_control.verifiers.qwen25_vl.classifier import (
    Qwen25VLBinaryAlignmentClassifier,
)


class QwenBinaryDecouplingTests(unittest.TestCase):
    def test_binary_prompt_module_owns_no_action_vocabulary(self):
        self.assertFalse(hasattr(binary_prompt, 'ACTION_NAMES'))
        self.assertFalse(hasattr(binary_prompt, 'ROUTING_STATUSES'))
        self.assertFalse(hasattr(binary_prompt, 'ROUTING_SYSTEM_PROMPT'))

    def test_binary_classifier_exposes_no_four_way_entrypoint(self):
        self.assertFalse(hasattr(
            Qwen25VLBinaryAlignmentClassifier,
            'verify_action',
        ))
        self.assertFalse(hasattr(
            Qwen25VLBinaryAlignmentClassifier,
            'classify_routing_candidate',
        ))


if __name__ == '__main__':
    unittest.main()
