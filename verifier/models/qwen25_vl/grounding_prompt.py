"""Single-object localization prompts for geometry-based routing."""

import json
from typing import Any, List, Mapping, Sequence

from PIL import Image

from .preprocessing import GROUNDING_ACTION_IMAGE_MODES


GROUNDING_PROMPT_PROTOCOLS = (
    'compact_json_v1',
    'single_object_json_v2',
)
DEFAULT_GROUNDING_PROMPT_PROTOCOL = 'compact_json_v1'
_COMPACT_GROUNDING_SYSTEM_PROMPT = (
    'You are an object localization model. Return exactly one JSON object and '
    'nothing else. Use this schema: '
    '{"bbox_2d":[x1,y1,x2,y2],"label":"object reference"}. '
    'The coordinates must be absolute xyxy coordinates on the exact image '
    'shown, using its stated width and height.'
)
_STRICT_GROUNDING_SYSTEM_PROMPT = (
    'You are a single-object localization engine. Your entire response must '
    'be exactly one compact JSON object on one line, with no Markdown fence '
    'and no explanation. Required schema: '
    '{"bbox_2d":[x1,y1,x2,y2],"label":"object reference"}. '
    'Protocol: the top level is one object, never a list; bbox_2d contains '
    'exactly four numeric values; label is a separate sibling field outside '
    'bbox_2d; output exactly one box and no additional keys. Coordinates are '
    'absolute xyxy values on the exact image shown and must satisfy '
    '0 <= x1 < x2 <= image width and '
    '0 <= y1 < y2 <= image height. If several instances exist, choose only '
    'the single instance that best matches the reference.'
)
GROUNDING_SYSTEM_PROMPTS = {
    'compact_json_v1': _COMPACT_GROUNDING_SYSTEM_PROMPT,
    'single_object_json_v2': _STRICT_GROUNDING_SYSTEM_PROMPT,
}
# Compatibility alias for callers that imported the former single prompt.
GROUNDING_SYSTEM_PROMPT = _STRICT_GROUNDING_SYSTEM_PROMPT


def build_reference_grounding_prompt(
        object_reference: str,
        image_size: Sequence[int],
        image_mode: str,
        prompt_protocol: str = DEFAULT_GROUNDING_PROMPT_PROTOCOL,
) -> str:
    """Ask Qwen to locate one reference without exposing candidate coordinates."""

    if image_mode not in GROUNDING_ACTION_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {GROUNDING_ACTION_IMAGE_MODES}, '
            f'got {image_mode!r}'
        )
    if prompt_protocol not in GROUNDING_PROMPT_PROTOCOLS:
        raise ValueError(
            f'prompt_protocol must be one of {GROUNDING_PROMPT_PROTOCOLS}, '
            f'got {prompt_protocol!r}'
        )
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError('image_size must be (width, height)')
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    reference = str(object_reference).strip() or 'the current object'
    image_note = (
        'The image is a clean source scene.'
        if image_mode == 'raw_image'
        else (
            'The image contains a red candidate rectangle. Treat the red '
            'rectangle only as a visual hint, not as ground truth.'
        )
    )
    prefix = (
        f'{image_note}\n'
        f'Image size: {width} x {height}.\n'
        f'Locate exactly one best-matching instance of "{reference}".\n'
    )
    if prompt_protocol == 'compact_json_v1':
        return prefix + 'Return only its bbox in the required JSON schema.'

    expected_json = (
        '{"bbox_2d":[x1,y1,x2,y2],"label":'
        f'{json.dumps(reference, ensure_ascii=False)}'
        '}'
    )
    return prefix + (
        'Select one instance only, even if several instances are visible.\n'
        'The bbox_2d array must contain exactly four numbers. Put the label '
        'outside that array as its sibling field.\n'
        f'Return exactly this one-line structure: {expected_json}\n'
        'Output:'
    )


def build_reference_grounding_messages(
        image: Image.Image,
        prompt: str,
        prompt_protocol: str = DEFAULT_GROUNDING_PROMPT_PROTOCOL,
) -> List[Mapping[str, Any]]:
    """Build the single-image localization conversation."""

    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    if prompt_protocol not in GROUNDING_PROMPT_PROTOCOLS:
        raise ValueError(
            f'prompt_protocol must be one of {GROUNDING_PROMPT_PROTOCOLS}, '
            f'got {prompt_protocol!r}'
        )
    return [
        {
            'role': 'system',
            'content': [
                {
                    'type': 'text',
                    'text': GROUNDING_SYSTEM_PROMPTS[prompt_protocol],
                },
            ],
        },
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        },
    ]
