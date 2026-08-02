"""CPU-only tests for the Qwen2.5-VL verifier backend."""

import unittest

from PIL import Image

from utils.coordinate_intervention import normalized_box_to_square_padding
from grounding_control.contracts import VerificationRequest
from grounding_control.coordinates import (
    center_pad_image,
    normalized_square_box_to_pixel_box,
    original_pixel_box_to_normalized_square_box,
)
from grounding_control.models.qwen25_vl.grounding_parser import (
    parse_reference_grounding_box,
    parse_reference_grounding_box_details,
)
from grounding_control.models.qwen25_vl.grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
    GROUNDING_SYSTEM_PROMPT,
    build_reference_grounding_prompt,
)
from grounding_control.models.qwen25_vl.preprocessing import qwen_smart_resize_size
from grounding_control.models.qwen25_vl.runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
)
from grounding_control.verifiers.qwen25_vl import (
    CandidateVerificationInput,
    build_binary_alignment_prompt,
    parse_binary_alignment_output,
    render_candidate_box,
    resize_crop_for_qwen,
)
from grounding_control.four_way.verifiers import (
    Qwen25VLGroundingGeometryClassifier,
    Qwen25VLVerifierBackend,
    route_from_grounding_geometry,
)
from grounding_control.four_way.verifiers.qwen25_vl import (
    GroundingActionInput,
    ROUTING_STATUSES,
    ROUTING_SYSTEM_PROMPT,
    build_routing_prompt,
    parse_routing_output,
    prepare_grounding_action_image,
)


class _FixedRunner:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.response


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

    def test_geometry_classifier_exposes_hard_action_without_fake_probs(self):
        classifier = Qwen25VLGroundingGeometryClassifier(
            runner=_FixedRunner(
                '{"bbox_2d":[10,10,30,30],"label":"object"}'
            )
        )
        output = classifier.classify_action(GroundingActionInput(
            image=Image.new('RGB', (100, 80), 'white'),
            object_reference='object',
            candidate_bbox_pixel_xyxy=(10, 10, 30, 30),
        ))
        self.assertEqual(output.predicted_action, 'no_action')
        self.assertIsNone(output.action_probabilities)
        self.assertEqual(
            output.metadata['probability_source'],
            'unavailable_geometry_hard_label',
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
        self.assertEqual(
            ROUTING_SYSTEM_PROMPT.count('The referenced object is present'),
            2,
        )
        self.assertNotIn('The referenced object is in img', ROUTING_SYSTEM_PROMPT)
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
        self.assertIn(
            'confidence in the chosen label, not P(aligned)',
            prompt,
        )
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
    def _request(**context_overrides):
        sample_context = {
            'image': Image.new('RGB', (10, 6), 'white'),
            **context_overrides,
        }
        return VerificationRequest(
            sample_id='sample-1',
            grounding_step=1,
            object_reference='baby',
            candidate_bbox=(0.1, 0.2, 0.3, 0.4),
            candidate_coordinate_text='',
            generated_ids=(),
            candidate_span=(0, 0),
            sample_context=sample_context,
        )

    @staticmethod
    def _candidate():
        return CandidateVerificationInput(
            image=Image.new('RGB', (10, 6), 'white'),
            object_reference='baby',
            candidate_bbox=(0.1, 0.2, 0.3, 0.4),
        )

    def test_verify_action_defaults_to_routing_four_way_bbox_image_only(self):
        runner = _FixedRunner(
            '{"status":"tighten","confidence":0.87}'
        )

        output = Qwen25VLVerifierBackend(runner=runner).verify_action(
            self._request()
        )

        self.assertEqual(output.predicted_action, 'tighten')
        self.assertEqual(output.confidence, 0.87)
        self.assertIsNone(output.action_probabilities)
        self.assertFalse(output.abstained)
        self.assertEqual(
            output.metadata['backend'],
            'qwen25_vl_routing_four_way_bbox_image_only',
        )
        self.assertEqual(output.metadata['image_mode'], 'bbox_image_only')
        image_blocks = [
            block
            for message in runner.messages
            for block in message['content']
            if block['type'] == 'image'
        ]
        self.assertEqual(len(image_blocks), 1)
        self.assertIn((255, 0, 0), set(image_blocks[0]['image'].getdata()))

    def test_malformed_routing_output_fails_open_as_abstention(self):
        output = Qwen25VLVerifierBackend(
            runner=_FixedRunner('not json'),
        ).verify_action(self._request())

        self.assertTrue(output.abstained)
        self.assertIsNone(output.predicted_action)
        self.assertEqual(output.confidence, 0.0)
        self.assertTrue(output.metadata['parse_failed'])
        self.assertIn('ValueError', output.error)

    def test_malformed_binary_output_fails_open_without_a_decision(self):
        lookup = Qwen25VLVerifierBackend(
            runner=_FixedRunner('not json'),
        ).verify_binary_alignment_candidate(self._candidate())

        self.assertIsNone(lookup.aligned)
        self.assertIsNone(lookup.confidence)
        self.assertTrue(lookup.metadata['parse_failed'])
        self.assertIn('ValueError', lookup.error)

    def test_verify_action_rejects_wrong_coordinate_system(self):
        backend = Qwen25VLVerifierBackend(
            runner=_FixedRunner(
                '{"status":"no_action","confidence":1.0}'
            )
        )

        with self.assertRaisesRegex(ValueError, 'center_padded_square'):
            backend.verify_action(self._request(
                coordinate_system='normalized_xyxy_on_original_image'
            ))

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

if __name__ == '__main__':
    unittest.main()
