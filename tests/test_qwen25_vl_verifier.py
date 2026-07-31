"""CPU-only tests for the Qwen2.5-VL verifier backend."""

import unittest

from PIL import Image
import torch

from utils.coordinate_intervention import normalized_box_to_square_padding
from verifier.backend import VerificationRequest
from verifier.backends.qwen25_vl import (
    CandidateVerificationInput,
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    GROUNDING_ACTION_OPTIONS,
    GROUNDING_PROMPT_PROTOCOLS,
    GROUNDING_SYSTEM_PROMPT,
    GroundingActionInput,
    LocalQwen25VLRunner,
    Qwen25VLGroundingActionClassifier,
    Qwen25VLGroundingGeometryClassifier,
    Qwen25VLVerifierBackend,
    ROUTING_STATUSES,
    SingleTokenOptionScores,
    STATUSES,
    build_binary_alignment_prompt,
    build_grounding_action_prompt,
    build_reference_grounding_prompt,
    build_routing_prompt,
    build_verification_prompt,
    center_pad_image,
    decide_from_option_scores,
    normalized_square_box_to_pixel_box,
    original_pixel_box_to_normalized_square_box,
    parse_binary_alignment_output,
    parse_reference_grounding_box,
    parse_reference_grounding_box_details,
    parse_routing_output,
    parse_verifier_output,
    prepare_grounding_action_image,
    qwen_smart_resize_size,
    render_candidate_box,
    resize_crop_for_qwen,
    route_from_grounding_geometry,
)
from verifier.backends.qwen25_vl.prompt import ROUTING_SYSTEM_PROMPT, SYSTEM_PROMPT


class _FixedRunner:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.response


class _FixedOptionRunner:
    min_pixels = 4 * 28 * 28
    max_pixels = 512 * 28 * 28

    def __init__(self, losses):
        self.losses = losses
        self.messages = None
        self.options = None

    def score_single_token_options(self, messages, options):
        self.messages = messages
        self.options = options
        return SingleTokenOptionScores(
            negative_log_likelihoods=dict(self.losses),
            token_ids={
                label: 100 + index
                for index, label in enumerate(options)
            },
        )


class _FakeBatch(dict):
    def __init__(self):
        input_ids = torch.tensor([[1, 2, 3]])
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return self


class _FakeTokenizer:
    _ids = {'A': 4, 'B': 5, 'C': 6, 'D': 7}

    def encode(self, text, add_special_tokens=False):
        return [self._ids[text]]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, *args, **kwargs):
        return _FakeBatch()


class _FakeModel:
    def __init__(self):
        self.call_count = 0

    def __call__(self, **kwargs):
        self.call_count += 1
        logits = torch.zeros((1, 3, 8), dtype=torch.float32)
        logits[0, -1, 4:8] = torch.tensor([0.0, 3.0, 2.0, 1.0])
        return type('Output', (), {'logits': logits})()


class QwenVerifierRenderingTests(unittest.TestCase):
    def test_tiny_crops_are_upscaled_to_a_56px_short_side(self):
        self.assertEqual(
            resize_crop_for_qwen(
                Image.new('RGB', (43, 15), 'white')
            ).size,
            (161, 56),
        )
        self.assertEqual(
            resize_crop_for_qwen(
                Image.new('RGB', (22, 32), 'white')
            ).size,
            (56, 82),
        )
        self.assertEqual(
            resize_crop_for_qwen(
                Image.new('RGB', (7, 4), 'white')
            ).size,
            (98, 56),
        )
        self.assertEqual(
            resize_crop_for_qwen(
                Image.new('RGB', (100, 80), 'white')
            ).size,
            (100, 80),
        )

    def test_landscape_odd_padding_matches_vocot_coordinates(self):
        image = Image.new('RGB', (7, 4), (10, 20, 30))
        padded, offset = center_pad_image(image)
        self.assertEqual(padded.size, (7, 7))
        self.assertEqual(offset, (0, 1))
        self.assertEqual(padded.getpixel((3, 0)), (122, 116, 104))
        self.assertEqual(padded.getpixel((3, 1)), (10, 20, 30))
        self.assertEqual(padded.getpixel((3, 4)), (10, 20, 30))
        self.assertEqual(padded.getpixel((3, 5)), (122, 116, 104))

        padded_box = normalized_box_to_square_padding(
            (0.0, 0.0, 1.0, 1.0), 7, 4
        )
        self.assertEqual(
            normalized_square_box_to_pixel_box(padded_box, 7),
            (0, 1, 6, 4),
        )

    def test_portrait_odd_padding_matches_vocot_coordinates(self):
        image = Image.new('RGB', (4, 7), (10, 20, 30))
        padded, offset = center_pad_image(image)
        self.assertEqual(padded.size, (7, 7))
        self.assertEqual(offset, (1, 0))
        self.assertEqual(padded.getpixel((0, 3)), (122, 116, 104))
        self.assertEqual(padded.getpixel((1, 3)), (10, 20, 30))
        self.assertEqual(padded.getpixel((4, 3)), (10, 20, 30))
        self.assertEqual(padded.getpixel((5, 3)), (122, 116, 104))

        padded_box = normalized_box_to_square_padding(
            (0.0, 0.0, 1.0, 1.0), 4, 7
        )
        self.assertEqual(
            normalized_square_box_to_pixel_box(padded_box, 7),
            (1, 0, 4, 6),
        )

    def test_square_image_and_full_box_have_no_offset(self):
        image = Image.new('RGB', (8, 8), (10, 20, 30))
        padded, offset = center_pad_image(image)
        self.assertEqual(padded.size, (8, 8))
        self.assertEqual(offset, (0, 0))
        self.assertEqual(
            normalized_square_box_to_pixel_box((0, 0, 1, 1), 8),
            (0, 0, 7, 7),
        )

    def test_candidate_is_not_padding_transformed_twice(self):
        image = Image.new('RGB', (7, 4), (10, 20, 30))
        padded_box = normalized_box_to_square_padding(
            (0.0, 0.0, 1.0, 1.0), 7, 4
        )
        rendered = render_candidate_box(
            image,
            padded_box,
            line_width=1,
        )
        self.assertEqual(rendered.padding_offset, (0, 1))
        self.assertEqual(rendered.pixel_bbox_xyxy, (0, 1, 6, 4))
        self.assertEqual(rendered.annotated_image.getpixel((0, 1)), (255, 0, 0))
        self.assertEqual(rendered.crop_image.size, (7, 4))
        self.assertEqual(rendered.crop_image.getpixel((0, 0)), (10, 20, 30))

    def test_original_pixel_box_converts_once_to_landscape_padded_square(self):
        padded_box = original_pixel_box_to_normalized_square_box(
            (0.0, 0.0, 7.0, 4.0),
            image_width=7,
            image_height=4,
        )
        self.assertEqual(padded_box, (0.0, 1 / 7, 1.0, 5 / 7))
        rendered = render_candidate_box(
            Image.new('RGB', (7, 4), 'white'),
            padded_box,
            line_width=1,
        )
        self.assertEqual(rendered.pixel_bbox_xyxy, (0, 1, 6, 4))

    def test_original_pixel_box_converts_once_to_portrait_padded_square(self):
        padded_box = original_pixel_box_to_normalized_square_box(
            (0.0, 0.0, 4.0, 7.0),
            image_width=4,
            image_height=7,
        )
        self.assertEqual(padded_box, (1 / 7, 0.0, 5 / 7, 1.0))
        rendered = render_candidate_box(
            Image.new('RGB', (4, 7), 'white'),
            padded_box,
            line_width=1,
        )
        self.assertEqual(rendered.pixel_bbox_xyxy, (1, 0, 4, 6))

    def test_grounding_action_uses_exact_qwen_resize_coordinate_frame(self):
        self.assertEqual(
            qwen_smart_resize_size(
                (100, 50),
                min_pixels=4 * 28 * 28,
                max_pixels=512 * 28 * 28,
            ),
            (112, 56),
        )
        prepared = prepare_grounding_action_image(
            Image.new('RGB', (100, 50), 'white'),
            (10, 5, 50, 25),
            min_pixels=4 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )
        self.assertEqual(prepared.original_size, (100, 50))
        self.assertEqual(prepared.model_size, (112, 56))
        self.assertEqual(
            prepared.candidate_bbox_model_xyxy,
            (11, 5, 56, 28),
        )
        self.assertNotIn(
            (255, 0, 0),
            set(prepared.clean_image.getdata()),
        )
        self.assertIn(
            (255, 0, 0),
            set(prepared.marked_image.getdata()),
        )


class QwenOptionLikelihoodTests(unittest.TestCase):
    def test_lowest_nll_decision_has_normalized_confidence_and_margin(self):
        decision = decide_from_option_scores(SingleTokenOptionScores(
            negative_log_likelihoods={
                'no_action': 2.0,
                'relocate': 0.5,
                'expand': 1.5,
                'tighten': 3.0,
            },
            token_ids={
                'no_action': 1,
                'relocate': 2,
                'expand': 3,
                'tighten': 4,
            },
        ))
        self.assertEqual(decision.label, 'relocate')
        self.assertAlmostEqual(decision.nll_margin, 1.0)
        self.assertAlmostEqual(
            sum(decision.normalized_probabilities.values()),
            1.0,
        )
        self.assertEqual(
            decision.confidence,
            decision.normalized_probabilities['relocate'],
        )

    def test_local_runner_scores_all_options_with_one_forward(self):
        runner = LocalQwen25VLRunner(model_path='unused')
        runner._processor = _FakeProcessor()
        runner._model = _FakeModel()
        runner._input_device = 'cpu'
        scores = runner.score_single_token_options(
            messages=[{'role': 'user', 'content': []}],
            options=GROUNDING_ACTION_OPTIONS,
        )
        self.assertEqual(runner._model.call_count, 1)
        self.assertEqual(
            min(
                scores.negative_log_likelihoods,
                key=scores.negative_log_likelihoods.__getitem__,
            ),
            'relocate',
        )
        self.assertEqual(
            scores.token_ids,
            {
                'no_action': 4,
                'relocate': 5,
                'expand': 6,
                'tighten': 7,
            },
        )

    def test_action_classifier_uses_raw_or_marked_image_with_same_bbox_text(self):
        runner = _FixedOptionRunner({
            'no_action': 2.0,
            'relocate': 0.5,
            'expand': 1.5,
            'tighten': 3.0,
        })
        classifier = Qwen25VLGroundingActionClassifier(runner=runner)
        candidate = GroundingActionInput(
            image=Image.new('RGB', (100, 50), 'white'),
            object_reference='baby',
            candidate_bbox_pixel_xyxy=(10, 5, 50, 25),
            sample_id='sample-1',
        )
        raw_lookup = classifier.classify(candidate, image_mode='raw_image')
        self.assertEqual(raw_lookup.status, 'relocate')
        self.assertEqual(raw_lookup.metadata['model_image_size'], [112, 56])
        self.assertEqual(
            raw_lookup.metadata['candidate_model_pixel_bbox_xyxy'],
            [11, 5, 56, 28],
        )
        raw_image = runner.messages[1]['content'][0]['image']
        self.assertNotIn((255, 0, 0), set(raw_image.getdata()))
        self.assertIn('Image size: 112 x 56', raw_lookup.metadata['prompt'])
        self.assertIn(
            'Candidate bbox (absolute xyxy): [11, 5, 56, 28]',
            raw_lookup.metadata['prompt'],
        )

        marked_lookup = classifier.classify(
            candidate,
            image_mode='bbox_image',
        )
        marked_image = runner.messages[1]['content'][0]['image']
        self.assertIn((255, 0, 0), set(marked_image.getdata()))
        self.assertEqual(
            raw_lookup.metadata['candidate_model_pixel_bbox_xyxy'],
            marked_lookup.metadata['candidate_model_pixel_bbox_xyxy'],
        )
        self.assertEqual(runner.options, GROUNDING_ACTION_OPTIONS)

    def test_action_prompt_exposes_only_one_fixed_option_question(self):
        prompt = build_grounding_action_prompt(
            object_reference='baby',
            candidate_bbox_xyxy=(11, 5, 56, 28),
            image_size=(112, 56),
            image_mode='raw_image',
        )
        self.assertIn('Object reference: "baby"', prompt)
        self.assertTrue(prompt.endswith('Answer:'))


class QwenGroundingGeometryTests(unittest.TestCase):
    def test_geometry_router_distinguishes_all_four_actions(self):
        self.assertEqual(
            route_from_grounding_geometry(
                (0, 0, 100, 100),
                (5, 5, 95, 95),
            ).action,
            'no_action',
        )
        expand = route_from_grounding_geometry(
            (25, 25, 75, 75),
            (0, 0, 100, 100),
        )
        self.assertEqual(expand.action, 'expand')
        self.assertEqual(expand.candidate_coverage_by_grounding, 1.0)
        self.assertEqual(expand.grounding_coverage_by_candidate, 0.25)

        tighten = route_from_grounding_geometry(
            (0, 0, 100, 100),
            (25, 25, 75, 75),
        )
        self.assertEqual(tighten.action, 'tighten')
        self.assertEqual(tighten.candidate_coverage_by_grounding, 0.25)
        self.assertEqual(tighten.grounding_coverage_by_candidate, 1.0)

        self.assertEqual(
            route_from_grounding_geometry(
                (0, 0, 40, 40),
                (60, 60, 100, 100),
            ).action,
            'relocate',
        )

    def test_grounding_parser_accepts_qwen_formats_and_rejects_bad_boxes(self):
        self.assertEqual(
            parse_reference_grounding_box(
                '{"bbox_2d":[11,5,56,28],"label":"baby"}',
                (112, 56),
            ),
            (11.0, 5.0, 56.0, 28.0),
        )
        self.assertEqual(
            parse_reference_grounding_box(
                '<|box_start|>(11,5),(56,28)<|box_end|>',
                (112, 56),
            ),
            (11.0, 5.0, 56.0, 28.0),
        )
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            parse_reference_grounding_box(
                '[{"bbox_2d":[1,1,2,2]},{"bbox_2d":[3,3,4,4]}]',
                (10, 10),
            )
        with self.assertRaisesRegex(ValueError, 'outside'):
            parse_reference_grounding_box(
                '{"bbox_2d":[1,1,20,20]}',
                (10, 10),
            )

    def test_grounding_parser_audits_one_pixel_boundary_clipping(self):
        parsed = parse_reference_grounding_box_details(
            '{"bbox_2d":[-1,-1,113,57],"label":"baby"}',
            (112, 56),
        )
        self.assertEqual(parsed.raw_box, (-1.0, -1.0, 113.0, 57.0))
        self.assertEqual(parsed.box, (0.0, 0.0, 112.0, 56.0))
        self.assertTrue(parsed.boundary_clipped)
        self.assertEqual(
            parsed.clipped_sides,
            ('left', 'top', 'right', 'bottom'),
        )
        self.assertEqual(parsed.boundary_tolerance_pixels, 1.0)
        with self.assertRaisesRegex(ValueError, 'beyond the 1-pixel tolerance'):
            parse_reference_grounding_box(
                '{"bbox_2d":[-1.01,0,56,28],"label":"baby"}',
                (112, 56),
            )

    def test_grounding_prompt_does_not_expose_candidate_coordinates(self):
        prompt = build_reference_grounding_prompt(
            'baby',
            (112, 56),
            'raw_image',
            prompt_protocol='single_object_json_v2',
        )
        self.assertIn('Locate exactly one best-matching instance of "baby"', prompt)
        self.assertIn('112 x 56', prompt)
        self.assertIn(
            'bbox_2d array must contain exactly four numbers',
            prompt,
        )
        self.assertIn(
            '{"bbox_2d":[x1,y1,x2,y2],"label":"baby"}',
            prompt,
        )
        self.assertNotIn('candidate bbox', prompt.lower())
        self.assertIn('top level is one object, never a list', GROUNDING_SYSTEM_PROMPT)
        self.assertEqual(
            DEFAULT_GROUNDING_PROMPT_PROTOCOL,
            'compact_json_v1',
        )
        self.assertEqual(
            GROUNDING_PROMPT_PROTOCOLS,
            ('compact_json_v1', 'single_object_json_v2'),
        )
        compact_prompt = build_reference_grounding_prompt(
            'baby',
            (112, 56),
            'raw_image',
            prompt_protocol='compact_json_v1',
        )
        self.assertNotIn('Select one instance only', compact_prompt)
        self.assertTrue(
            compact_prompt.endswith(
                'Return only its bbox in the required JSON schema.'
            )
        )

    def test_grounding_classifier_generates_box_then_routes_geometry(self):
        runner = _FixedRunner(
            '{"bbox_2d":[11,5,56,28],"label":"baby"}'
        )
        classifier = Qwen25VLGroundingGeometryClassifier(runner=runner)
        lookup = classifier.classify(GroundingActionInput(
            image=Image.new('RGB', (100, 50), 'white'),
            object_reference='baby',
            candidate_bbox_pixel_xyxy=(10, 5, 50, 25),
            sample_id='sample-1',
        ))
        self.assertEqual(lookup.status, 'no_action')
        self.assertIsNone(lookup.confidence)
        self.assertFalse(lookup.metadata['parse_failed'])
        self.assertEqual(
            lookup.metadata['prompt_protocol'],
            'compact_json_v1',
        )
        self.assertEqual(
            lookup.metadata['grounding_box_model_pixel_xyxy'],
            [11.0, 5.0, 56.0, 28.0],
        )
        self.assertEqual(lookup.metadata['geometry']['iou'], 1.0)

    def test_grounding_classifier_preserves_failed_raw_response(self):
        lookup = Qwen25VLGroundingGeometryClassifier(
            runner=_FixedRunner('I cannot locate it')
        ).classify(GroundingActionInput(
            image=Image.new('RGB', (100, 50), 'white'),
            object_reference='baby',
            candidate_bbox_pixel_xyxy=(10, 5, 50, 25),
        ))
        self.assertIsNone(lookup.status)
        self.assertTrue(lookup.metadata['parse_failed'])
        self.assertEqual(lookup.metadata['raw_response'], 'I cannot locate it')
        self.assertIn('ValueError', lookup.error)

    def test_grounding_classifier_logs_raw_and_clipped_boundary_box(self):
        lookup = Qwen25VLGroundingGeometryClassifier(
            runner=_FixedRunner(
                '{"bbox_2d":[-1,0,113,56],"label":"baby"}'
            )
        ).classify(GroundingActionInput(
            image=Image.new('RGB', (100, 50), 'white'),
            object_reference='baby',
            candidate_bbox_pixel_xyxy=(0, 0, 100, 50),
        ))
        self.assertEqual(lookup.status, 'no_action')
        self.assertFalse(lookup.metadata['parse_failed'])
        self.assertEqual(
            lookup.metadata['grounding_box_raw_model_pixel_xyxy'],
            [-1.0, 0.0, 113.0, 56.0],
        )
        self.assertEqual(
            lookup.metadata['grounding_box_model_pixel_xyxy'],
            [0.0, 0.0, 112.0, 56.0],
        )
        self.assertTrue(lookup.metadata['grounding_boundary_clipped'])
        self.assertEqual(
            lookup.metadata['grounding_boundary_clipped_sides'],
            ['left', 'right'],
        )
        self.assertEqual(
            lookup.metadata['grounding_boundary_tolerance_pixels'],
            1.0,
        )


class QwenVerifierParsingTests(unittest.TestCase):
    def test_binary_alignment_parser(self):
        parsed = parse_binary_alignment_output(
            '{"aligned":false,"confidence":0.91}'
        )
        self.assertFalse(parsed.aligned)
        self.assertEqual(parsed.confidence, 0.91)
        with self.assertRaises(ValueError):
            parse_binary_alignment_output(
                '{"aligned":"false","confidence":0.91}'
            )

    def test_routing_four_way_parser_and_prompt(self):
        parsed = parse_routing_output(
            '{"status":"relocate","confidence":0.91}'
        )
        self.assertEqual(parsed.status, 'relocate')
        self.assertEqual(parsed.confidence, 0.91)
        prompt = build_routing_prompt(
            'baby',
            (0.1, 0.2, 0.3, 0.4),
            image_mode='bbox_image_only',
        )
        self.assertIn('four defined actions', prompt)
        self.assertIn(
            'candidate region better match the object reference',
            prompt,
        )
        self.assertNotIn('Ignore every occurrence', prompt)
        self.assertIn('"baby"', prompt)
        self.assertEqual(
            ROUTING_STATUSES,
            ('no_action', 'relocate', 'expand', 'tighten'),
        )
        self.assertIn('You are a judge for visual grounding', ROUTING_SYSTEM_PROMPT)
        for status in ROUTING_STATUSES:
            self.assertIn(status, ROUTING_SYSTEM_PROMPT)
        with self.assertRaises(ValueError):
            parse_routing_output(
                '{"status":"wrong_object","confidence":0.91}'
            )
        with self.assertRaises(ValueError):
            parse_routing_output(
                '{"status":"aligned","confidence":0.91}'
            )
        crop_prompt = build_routing_prompt(
            'baby',
            (0.1, 0.2, 0.3, 0.4),
            image_mode='crop_only',
        )
        self.assertIn(
            'candidate visual region currently associated with the object '
            'reference',
            crop_prompt,
        )
        self.assertIn(
            'image correspond to the object reference more accurately',
            crop_prompt,
        )

    def test_binary_prompt_only_asks_alignment(self):
        prompt = build_binary_alignment_prompt('baby')
        self.assertIn('Does the visual content', prompt)
        self.assertIn('"baby"', prompt)
        for subtype in (
            'wrong_object',
            'partial_coverage',
            'ambiguous',
            'unsupported',
        ):
            self.assertNotIn(subtype, prompt)

    def test_marked_plus_crop_prompt_rejects_full_scene_leakage(self):
        prompt = build_binary_alignment_prompt(
            'baby',
            image_mode='marked_plus_crop',
        )
        self.assertIn('exact crop of the pixels inside', prompt)
        self.assertIn('Ignore every occurrence', prompt)
        self.assertIn('outside the red rectangle', prompt)
        self.assertIn('only evidence allowed', prompt)

    def test_bbox_image_only_prompt_rejects_full_scene_leakage(self):
        prompt = build_binary_alignment_prompt(
            'baby',
            image_mode='bbox_image_only',
        )
        self.assertIn('INSIDE the red rectangle', prompt)
        self.assertIn('outside the red rectangle', prompt)
        self.assertIn('Only the visual content inside', prompt)
        self.assertNotIn('Image 2', prompt)

    def test_all_five_statuses_map_to_legal_results(self):
        expected = {
            'aligned': ('aligned', 'none'),
            'wrong_object': ('misaligned', 'wrong_object'),
            'partial_coverage': ('misaligned', 'partial_coverage'),
            'ambiguous': ('uncertain', 'ambiguous'),
            'unsupported': ('misaligned', 'unsupported'),
        }
        for status in STATUSES:
            parsed = parse_verifier_output(
                f'```json\n{{"status":"{status}","confidence":0.75}}\n```'
            )
            self.assertEqual(
                (parsed.result.verdict, parsed.result.reason),
                expected[status],
            )
            self.assertEqual(parsed.result.confidence, 0.75)

    def test_invalid_output_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_verifier_output('{"status":"other","confidence":0.5}')
        with self.assertRaises(ValueError):
            parse_verifier_output('{"status":"aligned","confidence":1.5}')
        with self.assertRaises(ValueError):
            parse_verifier_output('not json')

    def test_system_prompt_defines_status_and_confidence_schema(self):
        prompt = build_verification_prompt(
            'Find the tissue box',
            (0.1, 0.2, 0.3, 0.4),
        )
        for status in STATUSES:
            self.assertIn(status, SYSTEM_PROMPT)
        self.assertIn('"status"', SYSTEM_PROMPT)
        self.assertIn('"confidence"', SYSTEM_PROMPT)
        self.assertIn('number from 0.0 to 1.0', SYSTEM_PROMPT)
        self.assertIn('Image 1 above', prompt)
        self.assertIn('Image 2 above', prompt)
        self.assertIn('Which one of the five defined statuses', prompt)
        self.assertNotIn('Original question', prompt)

    def test_five_way_prompts_match_the_selected_image_mode(self):
        crop_prompt = build_verification_prompt(
            'baby',
            (0.1, 0.2, 0.3, 0.4),
            image_mode='crop_only',
        )
        self.assertIn('image above is the exact border-free crop', crop_prompt)
        self.assertNotIn('red rectangle', crop_prompt)

        bbox_prompt = build_verification_prompt(
            'baby',
            (0.1, 0.2, 0.3, 0.4),
            image_mode='bbox_image_only',
        )
        self.assertIn('image above is the complete source scene', bbox_prompt)
        self.assertIn('inside the red rectangle', bbox_prompt)
        self.assertNotIn('Image 2', bbox_prompt)

        combined_prompt = build_verification_prompt(
            'baby',
            (0.1, 0.2, 0.3, 0.4),
            image_mode='marked_plus_crop',
        )
        self.assertIn('Image 1 above is the complete source scene', combined_prompt)
        self.assertIn('Image 2 above is the exact border-free crop', combined_prompt)


class QwenVerifierRunnerConfigurationTests(unittest.TestCase):
    def test_runner_uses_bounded_default_visual_resolution(self):
        runner = LocalQwen25VLRunner(model_path='unused')
        self.assertEqual(runner.min_pixels, DEFAULT_MIN_PIXELS)
        self.assertEqual(runner.max_pixels, 512 * 28 * 28)
        self.assertEqual(runner.max_pixels, DEFAULT_MAX_PIXELS)

    def test_runner_rejects_invalid_pixel_bounds(self):
        with self.assertRaises(ValueError):
            LocalQwen25VLRunner(
                model_path='unused',
                min_pixels=100,
                max_pixels=99,
            )


class QwenVerifierBackendTests(unittest.TestCase):
    @staticmethod
    def _request():
        return VerificationRequest(
            sample_id='sample-1',
            grounding_step=1,
            object_reference='Find the tissue box',
            candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            candidate_coordinate_text='<coor>0.1,0.2,0.3,0.4</coor>',
            generated_ids=(1, 2, 3),
            candidate_span=(1, 2),
            sample_context={
                'image': Image.new('RGB', (10, 6), 'white'),
                'question': 'What color is the tissue box?',
            },
        )

    def test_backend_returns_unsupported_without_loading_qwen(self):
        runner = _FixedRunner(
            '{"status":"unsupported","confidence":0.9}'
        )
        lookup = Qwen25VLVerifierBackend(runner=runner).verify(
            self._request()
        )
        self.assertEqual(lookup.result.verdict, 'misaligned')
        self.assertEqual(lookup.result.reason, 'unsupported')
        self.assertEqual(lookup.result.confidence, 0.9)
        self.assertEqual(
            lookup.metadata['coordinate_system'],
            'normalized_xyxy_on_center_padded_square',
        )
        self.assertEqual(lookup.metadata['original_image_size'], [10, 6])
        self.assertEqual(lookup.metadata['padded_square_size'], 10)
        self.assertEqual(lookup.metadata['padding_offset'], [0, 2])
        self.assertEqual(lookup.metadata['sample_id'], 'sample-1')
        self.assertEqual(lookup.metadata['candidate_crop_size'], [2, 2])
        self.assertEqual(lookup.metadata['model_crop_size'], [56, 56])
        self.assertNotIn(
            'What color is the tissue box?',
            lookup.metadata['prompt'],
        )
        self.assertIsNotNone(runner.messages)
        self.assertEqual(
            [message['role'] for message in runner.messages],
            ['system', 'user'],
        )
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 2)
        marked_image = image_blocks[0]['image']
        crop_image = image_blocks[1]['image']
        self.assertEqual(marked_image.size, (10, 10))
        self.assertEqual(crop_image.size, (56, 56))
        self.assertIn((255, 0, 0), set(marked_image.getdata()))
        self.assertNotIn((255, 0, 0), set(crop_image.getdata()))

    def test_five_way_can_use_only_candidate_crop(self):
        runner = _FixedRunner(
            '{"status":"unsupported","confidence":0.9}'
        )
        lookup = Qwen25VLVerifierBackend(
            runner=runner,
        ).verify_candidate(
            CandidateVerificationInput(
                image=Image.new('RGB', (10, 6), 'white'),
                object_reference='baby',
                candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            ),
            image_mode='crop_only',
        )
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]['image'].size, (56, 56))
        self.assertEqual(lookup.metadata['image_mode'], 'crop_only')
        self.assertEqual(lookup.metadata['model_image_count'], 1)
        self.assertEqual(lookup.metadata['model_crop_size'], [56, 56])

    def test_five_way_can_use_only_marked_bbox_image(self):
        runner = _FixedRunner(
            '{"status":"unsupported","confidence":0.9}'
        )
        lookup = Qwen25VLVerifierBackend(
            runner=runner,
        ).verify_candidate(
            CandidateVerificationInput(
                image=Image.new('RGB', (10, 6), 'white'),
                object_reference='baby',
                candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            ),
            image_mode='bbox_image_only',
        )
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]['image'].size, (10, 10))
        self.assertIn((255, 0, 0), set(image_blocks[0]['image'].getdata()))
        self.assertEqual(lookup.metadata['image_mode'], 'bbox_image_only')
        self.assertEqual(lookup.metadata['model_image_count'], 1)
        self.assertIsNone(lookup.metadata['model_crop_size'])

    def test_routing_four_way_uses_action_statuses(self):
        runner = _FixedRunner(
            '{"status":"expand","confidence":0.88}'
        )
        lookup = Qwen25VLVerifierBackend(
            runner=runner,
        ).classify_routing_candidate(
            CandidateVerificationInput(
                image=Image.new('RGB', (10, 6), 'white'),
                object_reference='baby',
                candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            ),
            image_mode='bbox_image_only',
        )
        self.assertEqual(lookup.status, 'expand')
        self.assertEqual(lookup.confidence, 0.88)
        self.assertEqual(lookup.metadata['model_image_count'], 1)
        self.assertEqual(
            lookup.metadata['backend'],
            'qwen25_vl_routing_four_way_bbox_image_only',
        )

    def test_binary_alignment_uses_only_candidate_crop(self):
        runner = _FixedRunner(
            '{"aligned":false,"confidence":0.92}'
        )
        lookup = Qwen25VLVerifierBackend(
            runner=runner,
        ).verify_binary_alignment_candidate(
            CandidateVerificationInput(
                image=Image.new('RGB', (10, 6), 'white'),
                object_reference='baby',
                candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            )
        )
        self.assertFalse(lookup.aligned)
        self.assertEqual(lookup.confidence, 0.92)
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]['image'].size, (56, 56))
        self.assertEqual(lookup.metadata['candidate_crop_size'], [2, 2])
        self.assertEqual(lookup.metadata['model_crop_size'], [56, 56])
        self.assertEqual(lookup.metadata['binary_image_mode'], 'crop_only')
        self.assertEqual(lookup.metadata['model_image_count'], 1)

    def test_binary_alignment_can_add_marked_scene_before_same_crop(self):
        runner = _FixedRunner(
            '{"aligned":false,"confidence":0.92}'
        )
        lookup = Qwen25VLVerifierBackend(
            runner=runner,
        ).verify_binary_alignment_candidate(
            CandidateVerificationInput(
                image=Image.new('RGB', (10, 6), 'white'),
                object_reference='baby',
                candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            ),
            image_mode='marked_plus_crop',
        )
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 2)
        self.assertEqual(image_blocks[0]['image'].size, (10, 10))
        self.assertEqual(image_blocks[1]['image'].size, (56, 56))
        self.assertEqual(
            lookup.metadata['binary_image_mode'],
            'marked_plus_crop',
        )
        self.assertEqual(lookup.metadata['model_image_count'], 2)

    def test_binary_alignment_can_use_only_marked_bbox_image(self):
        runner = _FixedRunner(
            '{"aligned":false,"confidence":0.92}'
        )
        lookup = Qwen25VLVerifierBackend(
            runner=runner,
        ).verify_binary_alignment_candidate(
            CandidateVerificationInput(
                image=Image.new('RGB', (10, 6), 'white'),
                object_reference='baby',
                candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            ),
            image_mode='bbox_image_only',
        )
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]['image'].size, (10, 10))
        self.assertIn((255, 0, 0), set(image_blocks[0]['image'].getdata()))
        self.assertEqual(
            lookup.metadata['binary_image_mode'],
            'bbox_image_only',
        )
        self.assertEqual(lookup.metadata['model_image_count'], 1)
        self.assertIsNone(lookup.metadata['model_crop_size'])

    def test_parse_failure_fails_open_as_uncertain(self):
        lookup = Qwen25VLVerifierBackend(
            runner=_FixedRunner('I cannot decide'),
        ).verify(self._request())
        self.assertEqual(lookup.result.verdict, 'uncertain')
        self.assertEqual(lookup.result.confidence, 0.0)
        self.assertTrue(lookup.metadata['parse_failed'])
        self.assertIn('ValueError', lookup.error)

    def test_backend_rejects_wrong_coordinate_system(self):
        request = self._request()
        wrong_context = dict(request.sample_context)
        wrong_context['coordinate_system'] = 'normalized_xyxy_on_original_image'
        request = VerificationRequest(
            sample_id=request.sample_id,
            grounding_step=request.grounding_step,
            object_reference=request.object_reference,
            candidate_bbox=request.candidate_bbox,
            candidate_coordinate_text=request.candidate_coordinate_text,
            generated_ids=request.generated_ids,
            candidate_span=request.candidate_span,
            sample_context=wrong_context,
        )
        with self.assertRaisesRegex(ValueError, 'center_padded_square'):
            Qwen25VLVerifierBackend(
                runner=_FixedRunner(
                    '{"status":"aligned","confidence":1.0}'
                )
            ).verify(request)


if __name__ == '__main__':
    unittest.main()
