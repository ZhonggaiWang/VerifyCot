"""Unit tests for the versioned Grounder worker payload."""

import unittest

from grounding_control.transport import (
    GROUNDER_OUTPUT_SCHEMA,
    parse_grounder_output,
    serialize_grounder_output,
)


class GrounderWireTests(unittest.TestCase):
    def test_available_round_trip_uses_only_canonical_fields(self):
        response = serialize_grounder_output(
            available=True,
            source='grounding_dino_grounder',
            bbox=(1.0, 2.0, 9.0, 8.0),
            image_size=(10, 10),
            confidence=0.75,
            error=None,
            metadata={'detector': 'dino'},
        )

        self.assertEqual(response, {
            'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
            'available': True,
            'source': 'grounding_dino_grounder',
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'bbox': [1.0, 2.0, 9.0, 8.0],
            'image_size': [10, 10],
            'confidence': 0.75,
            'error': None,
            'metadata': {'detector': 'dino'},
        })
        parsed = parse_grounder_output({
            'request_id': 'transport-field-is-allowed',
            **response,
        })
        self.assertEqual(parsed.bbox, (1.0, 2.0, 9.0, 8.0))

    def test_unavailable_requires_error_and_no_box_or_confidence(self):
        response = serialize_grounder_output(
            available=False,
            source='qwen25_vl_grounder',
            bbox=None,
            image_size=(10, 10),
            confidence=None,
            error='coordinate_parse_failed',
            metadata={'parse_failed': True},
        )
        parsed = parse_grounder_output(response)
        self.assertFalse(parsed.available)
        self.assertIsNone(parsed.bbox)
        self.assertEqual(parsed.error, 'coordinate_parse_failed')

    def test_legacy_bbox_key_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, 'bbox'):
            parse_grounder_output({
                'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
                'available': True,
                'source': 'legacy',
                'coordinate_system': 'absolute_xyxy_on_original_image',
                'bbox_original_pixel_xyxy': [1, 2, 9, 8],
                'image_size': [10, 10],
                'confidence': 0.5,
                'error': None,
                'metadata': {},
            })


if __name__ == '__main__':
    unittest.main()
