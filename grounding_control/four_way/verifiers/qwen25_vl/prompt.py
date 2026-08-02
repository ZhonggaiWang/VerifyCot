"""Prompts for binary and direct four-way Qwen2.5-VL verification."""

from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image

from ...contracts import ACTION_NAMES


# Historical Qwen code exposed this name.  Keep the alias, but use the public
# routing contract as the single source of truth for the four action labels.
ROUTING_STATUSES = ACTION_NAMES

ROUTING_SYSTEM_PROMPT = (
    'You are a judge for visual grounding. Based on the relationship between '
    'the object reference and the candidate region shown in the image, decide '
    'which one of the following four actions should be taken:\n'
    'relocate: The referenced object is not correctly located in the candidate '
    'region. The region either contains a different object or contains mainly '
    'background with no recognizable evidence for the reference. A new region '
    'must be located from scratch.\n'
    'expand: The referenced object is present, but the candidate region clips '
    'it or covers only an insufficient part. The region must be expanded or '
    'refined to cover the complete object.\n'
    'tighten: The referenced object is present, but the candidate region is too '
    'broad or contains other salient objects. The region must be tightened to '
    'uniquely localize the reference.\n'
    'no_action: The candidate region already correctly, sufficiently, and '
    'uniquely localizes the referenced object. Keep it unchanged and perform '
    'no corrective operation.\n'
    'Return exactly one JSON object with two keys. The "status" value must be '
    'exactly one of: no_action, relocate, expand, tighten. The "confidence" '
    'value must be a number from 0.0 to 1.0. Return no markdown or explanation.'
)


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
        image_mode: str = 'crop_only',
) -> str:
    """Ask only the minimal local reference--image alignment question."""

    if image_mode not in BINARY_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {BINARY_IMAGE_MODES}, got {image_mode!r}'
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
        image_mode: str = 'crop_only',
) -> List[Mapping[str, Any]]:
    """Build the explicitly selected binary image-input protocol."""

    if image_mode not in BINARY_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {BINARY_IMAGE_MODES}, got {image_mode!r}'
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
    else:  # marked_plus_crop
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
    system_prompt = protocol['system_prompt']
    return [
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text': system_prompt}
            ],
        },
        {
            'role': 'user',
            'content': image_content + [{'type': 'text', 'text': prompt}],
        },
    ]


def build_routing_prompt(
        object_reference: str,
        candidate_bbox: Sequence[float],
        image_mode: str = 'bbox_image_only',
) -> str:
    """Describe one image protocol and request a routing action status."""

    if image_mode not in BINARY_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {BINARY_IMAGE_MODES}, got {image_mode!r}'
        )
    reference = str(object_reference).strip() or 'the current object'
    if image_mode == 'crop_only':
        return (
            'The image above is the candidate visual region currently '
            'associated with the object reference.\n'
            f'Object reference: "{reference}"\n'
            'Which one of the four defined actions should be taken to make '
            'the image correspond to the object reference more accurately?'
        )
    if image_mode == 'bbox_image_only':
        return (
            'The image above is the complete source scene with one candidate '
            'region outlined by a red rectangle.\n'
            f'Object reference: "{reference}"\n'
            'Which one of the four defined actions should be taken to make '
            'the candidate region better match the object reference?'
        )
    return (
        'Image 1 above is the complete source scene with one candidate region '
        'outlined by a red rectangle. Image 2 above is the exact border-free '
        'crop of the content inside that rectangle.\n'
        f'Object reference: "{reference}"\n'
        'Which one of the four defined actions should be taken to make '
        'the candidate region better match the object reference?'
    )


def build_qwen_messages(
        annotated_image: Optional[Image.Image],
        crop_image: Optional[Image.Image],
        prompt: str,
        image_mode: str = 'marked_plus_crop',
        system_prompt: str = ROUTING_SYSTEM_PROMPT,
) -> List[Mapping[str, Any]]:
    """Create direct four-way messages for one selected image mode."""

    if image_mode not in BINARY_IMAGE_MODES:
        raise ValueError(
            f'image_mode must be one of {BINARY_IMAGE_MODES}, got {image_mode!r}'
        )
    if image_mode == 'crop_only':
        if crop_image is None:
            raise ValueError('crop_image is required for crop_only')
        image_content = [{'type': 'image', 'image': crop_image}]
    elif image_mode == 'bbox_image_only':
        if annotated_image is None:
            raise ValueError(
                'annotated_image is required for bbox_image_only'
            )
        image_content = [{'type': 'image', 'image': annotated_image}]
    else:
        if annotated_image is None:
            raise ValueError(
                'annotated_image is required for marked_plus_crop'
            )
        if crop_image is None:
            raise ValueError('crop_image is required for marked_plus_crop')
        image_content = [
            {'type': 'image', 'image': annotated_image},
            {'type': 'image', 'image': crop_image},
        ]
    return [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': system_prompt}],
        },
        {
            'role': 'user',
            'content': image_content + [{'type': 'text', 'text': prompt}],
        },
    ]
