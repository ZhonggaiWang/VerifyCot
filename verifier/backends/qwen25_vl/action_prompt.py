"""Prompts for four-way grounding actions scored as fixed options."""

from typing import Any, List, Mapping, Sequence, Tuple

from PIL import Image


GROUNDING_ACTION_IMAGE_MODES = ('raw_image', 'bbox_image')
GROUNDING_ACTION_OPTIONS = {
    'no_action': 'A',
    'relocate': 'B',
    'expand': 'C',
    'tighten': 'D',
}

GROUNDING_ACTION_SYSTEM_PROMPT = (
    'You are a visual grounding action classifier. First locate the object '
    'reference internally in the image. Do not output the location or any '
    'reasoning. Compare the candidate bounding box with the object you locate '
    'and choose exactly one action:\n'
    'A. no_action: The candidate bbox correctly, sufficiently, and uniquely '
    'localizes the referenced object.\n'
    'B. relocate: The candidate bbox localizes another object or background '
    'and must be replaced by a newly located region.\n'
    'C. expand: The candidate bbox contains the referenced object but covers '
    'only part of it and must be expanded.\n'
    'D. tighten: The candidate bbox contains the referenced object but is too '
    'broad or ambiguous and must be tightened.\n'
    'The answer is scored as one fixed option. Select A, B, C, or D only.'
)


def _integer_box(values: Sequence[int]) -> Tuple[int, int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('candidate bbox must have four integer coordinates')
    box = tuple(int(value) for value in values)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f'candidate bbox is empty: {box}')
    return box  # type: ignore[return-value]


def build_grounding_action_prompt(
        object_reference: str,
        candidate_bbox_xyxy: Sequence[int],
        image_size: Sequence[int],
        image_mode: str,
) -> str:
    """Describe the exact Qwen image-coordinate frame and request A/B/C/D."""

    if image_mode not in GROUNDING_ACTION_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {GROUNDING_ACTION_IMAGE_MODES}, '
            f'got {image_mode!r}'
        )
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError('image_size must be (width, height)')
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    x_min, y_min, x_max, y_max = _integer_box(candidate_bbox_xyxy)
    if not (
        0 <= x_min < x_max <= width
        and 0 <= y_min < y_max <= height
    ):
        raise ValueError(
            f'candidate bbox {(x_min, y_min, x_max, y_max)} is outside '
            f'{width}x{height}'
        )
    reference = str(object_reference).strip() or 'the current object'
    image_description = (
        'The image is the clean source image.'
        if image_mode == 'raw_image'
        else (
            'The image is the source image with the candidate bbox outlined '
            'by one red rectangle.'
        )
    )
    return (
        f'{image_description}\n'
        f'Object reference: "{reference}"\n'
        f'Image size: {width} x {height}\n'
        f'Candidate bbox (absolute xyxy): '
        f'[{x_min}, {y_min}, {x_max}, {y_max}]\n'
        f'Internally locate "{reference}", compare the candidate bbox with '
        'that object, and select the best action.\n'
        'Answer:'
    )


def build_grounding_action_messages(
        image: Image.Image,
        prompt: str,
) -> List[Mapping[str, Any]]:
    """Build a single-image chat whose next token is the option code."""

    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    return [
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': GROUNDING_ACTION_SYSTEM_PROMPT},
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
