"""CPU-only tests for the Qwen2.5-VL verifier backend."""

import unittest

from PIL import Image

from utils.coordinate_intervention import normalized_box_to_square_padding
from verifier.backend import VerificationRequest
from verifier.backends.qwen25_vl import (
    CandidateVerificationInput,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLVerifierBackend,
    ROUTING_STATUSES,
    STATUSES,
    build_binary_alignment_prompt,
    build_routing_prompt,
    build_verification_prompt,
    center_pad_image,
    normalized_square_box_to_pixel_box,
    original_pixel_box_to_normalized_square_box,
    parse_binary_alignment_output,
    parse_routing_output,
    parse_verifier_output,
    render_candidate_box,
    resize_crop_for_qwen,
)
from verifier.backends.qwen25_vl.prompt import ROUTING_SYSTEM_PROMPT, SYSTEM_PROMPT


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
