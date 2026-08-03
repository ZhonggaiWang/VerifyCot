"""Tests for the public binary pre-commit inference wrapper."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import model.load_model as load_model_module


class AlignmentRoutingInferTests(unittest.TestCase):
    def test_wrapper_builds_binary_policy_and_reuses_routing_validation(self):
        sentinel = {'status': 'ok'}
        with patch.object(
                load_model_module,
                'routing_infer',
                return_value=sentinel) as routed:
            result = load_model_module.alignment_routing_infer(
                model='model',
                preprocessor='preprocessor',
                image='image',
                verifier_backend='verifier',
                grounder_backend='grounder',
                reject_threshold=0.2,
                accept_threshold=0.8,
                sample_id='sample',
            )

        self.assertIs(result, sentinel)
        kwargs = routed.call_args.kwargs
        policy = kwargs['alignment_routing_policy']
        self.assertEqual(policy.reject_threshold, 0.2)
        self.assertEqual(policy.accept_threshold, 0.8)
        self.assertEqual(kwargs['missing_expert_policy'], 'fail_open')

    def test_wrapper_rejects_invalid_threshold_order_before_generation(self):
        with patch.object(load_model_module, 'routing_infer') as routed:
            with self.assertRaises(ValueError):
                load_model_module.alignment_routing_infer(
                    model='model',
                    preprocessor='preprocessor',
                    image='image',
                    verifier_backend='verifier',
                    grounder_backend='grounder',
                    reject_threshold=0.8,
                    accept_threshold=0.2,
                    sample_id='sample',
                )
        routed.assert_not_called()

    def test_wrapper_requires_explicit_raw_kind_for_proxy_thresholds(self):
        sentinel = {'status': 'ok'}
        with patch.object(
                load_model_module,
                'routing_infer',
                return_value=sentinel) as routed:
            result = load_model_module.alignment_routing_infer(
                model='model',
                preprocessor='preprocessor',
                image='image',
                verifier_backend='verifier',
                grounder_backend='grounder',
                reject_threshold=0.2,
                accept_threshold=0.8,
                alignment_score_kind='iou_proxy',
                sample_id='sample',
            )

        self.assertIs(result, sentinel)
        policy = routed.call_args.kwargs['alignment_routing_policy']
        self.assertEqual(policy.required_score_kind, 'iou_proxy')
        self.assertIsNone(policy.calibrator)


class RoutingInferTerminalAbstainTests(unittest.TestCase):
    @staticmethod
    def _terminal_event():
        return {
            'grounding_step': 1,
            'candidate_box': [0.7, 0.7, 0.9, 0.9],
            'verifier_output_schema': 'vocot_four_action_v1',
            'routing_decision': 'abstain',
            'router_action': 'routing_abstained',
            'candidate_refbind_uncommitted': True,
            'candidate_committed': False,
            'coordinate_committed': False,
            'terminal_uncommitted': True,
            'committed_coordinate_text': None,
            'committed_box': None,
            'committed_feature_will_be_injected_on_clean_replay': False,
        }

    def _run_with_controller_result(self, controller_result):
        class FakeController:
            def __init__(self, **kwargs):
                del kwargs

            def run(self, max_new_tokens, temperature):
                del max_new_tokens, temperature
                return controller_result

        tokenizer = SimpleNamespace(eos_token_id=2)
        model = SimpleNamespace(
            boc_token_id=1000,
            eoc_token_id=1001,
            last_bound_boxes=[],
        )
        with patch(
                'grounding_control.four_way.'
                'FourWayPrecommitGroundingController',
                FakeController):
            return load_model_module.routing_infer(
                model=model,
                preprocessor=SimpleNamespace(tokenizer=tokenizer),
                image='image',
                verifier_backend='verifier',
                grounder_backend=None,
                sample_id='sample',
            )

    def test_terminal_abstain_returns_auditable_uncommitted_event(self):
        event = self._terminal_event()
        result = self._run_with_controller_result(SimpleNamespace(
            response='Find the cup ',
            generated_ids=[7, 8, 9],
            status='routing_abstained',
            events=[event],
        ))

        self.assertEqual(result['status'], 'routing_abstained')
        self.assertEqual(result['boxes'], [])
        self.assertEqual(result['bound_boxes'], [])
        self.assertEqual(result['events'], [event])
        self.assertEqual(
            event['verification'],
            'candidate_uncommitted_before_refbind',
        )

    def test_uncommitted_event_cannot_bypass_normal_postvalidation(self):
        with self.assertRaisesRegex(
                RuntimeError,
                'only as the final routing_abstained decision'):
            self._run_with_controller_result(SimpleNamespace(
                response='Find the cup ',
                generated_ids=[7, 8, 9],
                status='ok',
                events=[self._terminal_event()],
            ))


if __name__ == '__main__':
    unittest.main()
