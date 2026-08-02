"""Prompts owned by the standalone Qwen binary verifier path.

This module deliberately imports no four-way action contract or routing
prompt.  The binary verifier can therefore be deployed without loading the
legacy action-classification implementation.
"""

from typing import Any, List, Mapping, Optional

from PIL import Image


BINARY_ALIGNMENT_SYSTEM_PROMPT = (
    'Judge whether the candidate image aligns with the given object reference. '
    'Return JSON only.'
)

BINARY_MARKED_PLUS_CROP_SYSTEM_PROMPT = (
    'You are a strict local visual grounding verifier. Image 1 is the complete '
    'scene and contains one red candidate rectangle. Image 2 is the exact '
    'border-free crop of the visual content inside that red rectangle. Judge '
    'only whether the region inside the red rectangle, confirmed by Image 2, '
    'aligns with the object reference. An object elsewhere in Image 1 is '
    'irrelevant and must never make the candidate aligned. Return JSON only.'
)

BINARY_BBOX_IMAGE_ONLY_SYSTEM_PROMPT = (
    'You are a strict local visual grounding verifier. The image is the '
    'complete scene and contains one red candidate rectangle. Judge only '
    'whether the visual content inside that red rectangle aligns with the '
    'object reference. An object elsewhere outside the red rectangle is '
    'irrelevant and must never make the candidate aligned. Return JSON only.'
)

BINARY_IMAGE_PROTOCOLS = {
    'crop_only': {
        'system_prompt': BINARY_ALIGNMENT_SYSTEM_PROMPT,
        'model_image_count': 1,
        'uses_bbox_image': False,
        'uses_crop': True,
    },
    'bbox_image_only': {
        'system_prompt': BINARY_BBOX_IMAGE_ONLY_SYSTEM_PROMPT,
        'model_image_count': 1,
        'uses_bbox_image': True,
        'uses_crop': False,
    },
    'marked_plus_crop': {
        'system_prompt': BINARY_MARKED_PLUS_CROP_SYSTEM_PROMPT,
        'model_image_count': 2,
        'uses_bbox_image': True,
        'uses_crop': True,
    },
}
BINARY_IMAGE_MODES = tuple(BINARY_IMAGE_PROTOCOLS)


def build_binary_alignment_prompt(
        object_reference: str,
        image_mode: str = 'crop_only') -> str:
    """Ask only the local reference--candidate alignment question."""

    if image_mode not in BINARY_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {BINARY_IMAGE_MODES}, '
            f'got {image_mode!r}'
        )
    reference = str(object_reference).strip() or 'the current object'
    if image_mode == 'bbox_image_only':
        return (
            f'Object reference: "{reference}"\n'
            'The image is the complete scene with the candidate region '
            'outlined by one red rectangle.\n\n'
            f'Decide whether the visual content INSIDE the red rectangle '
            f'correctly and sufficiently localizes "{reference}".\n'
            'Do not decide whether the complete scene contains the referenced '
            'object. Ignore every occurrence of the referenced object outside '
            'the red rectangle. Only the visual content inside the red '
            'rectangle is evidence for this decision.\n'
            'Set "aligned" to true only if the red-box region itself shows and '
            'sufficiently localizes the referenced object. Set it to false if '
            'the region shows another object, background, only an insufficient '
            'part, or a non-specific/ambiguous region.\n'
            'Return exactly one JSON object with two keys: "aligned" as the '
            'JSON boolean true or false, and "confidence" as a number from '
            '0.5 to 1.0 measuring confidence that the emitted "aligned" '
            'boolean is correct. "confidence" is confidence in the chosen '
            'label, not P(aligned).'
        )
    if image_mode == 'marked_plus_crop':
        return (
            f'Object reference: "{reference}"\n'
            'Image 1: the complete scene with the candidate region outlined '
            'by one red rectangle.\n'
            'Image 2: the exact crop of the pixels inside that red rectangle.\n\n'
            f'Decide whether the content INSIDE the red rectangle, as shown '
            f'again in Image 2, correctly and sufficiently localizes '
            f'"{reference}".\n'
            'Do not decide whether the complete scene contains the referenced '
            'object. Ignore every occurrence of the referenced object outside '
            'the red rectangle. The red-box region and its crop are the only '
            'evidence allowed for the decision.\n'
            'Set "aligned" to true only if that candidate region itself shows '
            'and sufficiently localizes the referenced object. Set it to false '
            'if the region shows another object, background, only an '
            'insufficient part, or a non-specific/ambiguous region.\n'
            'Return exactly one JSON object with two keys: "aligned" as the '
            'JSON boolean true or false, and "confidence" as a number from '
            '0.5 to 1.0 measuring confidence that the emitted "aligned" '
            'boolean is correct. "confidence" is confidence in the chosen '
            'label, not P(aligned).'
        )
    return (
        f'Object reference: "{reference}"\n'
        f'Does the visual content in the candidate image correctly and '
        f'sufficiently align with "{reference}"?\n'
        'Set "aligned" to true only when the candidate image itself shows and '
        'sufficiently localizes that object; otherwise set it to false.\n'
        'Return exactly one JSON object with two keys: "aligned" as the JSON '
        'boolean true or false, and "confidence" as a number from 0.5 to 1.0 '
        'measuring confidence that the emitted "aligned" boolean is correct. '
        '"confidence" is confidence in the chosen label, not P(aligned).'
    )


def build_binary_alignment_messages(
        crop_image: Optional[Image.Image],
        prompt: str,
        annotated_image: Optional[Image.Image] = None,
        image_mode: str = 'crop_only') -> List[Mapping[str, Any]]:
    """Build the explicitly selected binary image-input protocol."""

    if image_mode not in BINARY_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {BINARY_IMAGE_MODES}, '
            f'got {image_mode!r}'
        )
    protocol = BINARY_IMAGE_PROTOCOLS[image_mode]
    if protocol['uses_bbox_image'] and annotated_image is None:
        raise ValueError(
            f'annotated_image is required for image_mode={image_mode!r}'
        )
    if protocol['uses_crop'] and crop_image is None:
        raise ValueError(
            f'crop_image is required for image_mode={image_mode!r}'
        )

    if image_mode == 'crop_only':
        image_content = [
            {
                'type': 'text',
                'text': 'Candidate image: crop from inside the candidate box.',
            },
            {'type': 'image', 'image': crop_image},
        ]
    elif image_mode == 'bbox_image_only':
        image_content = [
            {
                'type': 'text',
                'text': (
                    'Candidate image: complete scene with the decision region '
                    'outlined in red. Judge only the content inside the red '
                    'rectangle and ignore matching objects outside it.'
                ),
            },
            {'type': 'image', 'image': annotated_image},
        ]
    else:
        image_content = [
            {
                'type': 'text',
                'text': (
                    'Image 1 — context only: complete scene with the candidate '
                    'region outlined in red. Objects outside the red rectangle '
                    'must be ignored.'
                ),
            },
            {'type': 'image', 'image': annotated_image},
            {
                'type': 'text',
                'text': (
                    'Image 2 — decision region: exact border-free crop of the '
                    'pixels inside the red rectangle in Image 1.'
                ),
            },
            {'type': 'image', 'image': crop_image},
        ]
    return [
        {
            'role': 'system',
            'content': [{
                'type': 'text',
                'text': protocol['system_prompt'],
            }],
        },
        {
            'role': 'user',
            'content': image_content + [{'type': 'text', 'text': prompt}],
        },
    ]


__all__ = [
    'BINARY_IMAGE_MODES',
    'BINARY_IMAGE_PROTOCOLS',
    'build_binary_alignment_messages',
    'build_binary_alignment_prompt',
]
