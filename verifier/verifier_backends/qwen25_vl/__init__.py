"""Qwen2.5-VL binary and four-action verifier implementations.

Reusable Qwen model execution and object-to-box prediction live under
``verifier.models.qwen25_vl``.  This package contains only candidate-region
verification and the Qwen-box-plus-geometry verifier composition.
"""

from .direct import (
    BinaryAlignmentLookup,
    CandidateVerificationInput,
    Qwen25VLVerifierBackend,
    RoutingClassificationLookup,
)
from .grounding_geometry import (
    GroundingGeometryLookup,
    Qwen25VLGroundingGeometryClassifier,
)
from .grounding_input import (
    GroundingActionInput,
    PreparedGroundingActionImage,
    prepare_grounding_action_image,
)
from .parser import (
    ParsedBinaryAlignmentOutput,
    ParsedRoutingOutput,
    parse_binary_alignment_output,
    parse_routing_output,
)
from .prompt import (
    BINARY_IMAGE_MODES,
    BINARY_IMAGE_PROTOCOLS,
    ROUTING_STATUSES,
    ROUTING_SYSTEM_PROMPT,
    build_binary_alignment_messages,
    build_binary_alignment_prompt,
    build_qwen_messages,
    build_routing_prompt,
)
from .rendering import (
    COORDINATE_SYSTEM,
    DEFAULT_QWEN_CROP_MIN_SIDE,
    RenderedCandidate,
    center_pad_image,
    normalized_square_box_to_pixel_box,
    original_pixel_box_to_normalized_square_box,
    render_candidate_box,
    resize_crop_for_qwen,
)

__all__ = [
    'BINARY_IMAGE_MODES',
    'BINARY_IMAGE_PROTOCOLS',
    'BinaryAlignmentLookup',
    'COORDINATE_SYSTEM',
    'CandidateVerificationInput',
    'DEFAULT_QWEN_CROP_MIN_SIDE',
    'GroundingActionInput',
    'GroundingGeometryLookup',
    'ParsedBinaryAlignmentOutput',
    'ParsedRoutingOutput',
    'PreparedGroundingActionImage',
    'Qwen25VLGroundingGeometryClassifier',
    'Qwen25VLVerifierBackend',
    'ROUTING_STATUSES',
    'ROUTING_SYSTEM_PROMPT',
    'RenderedCandidate',
    'RoutingClassificationLookup',
    'build_binary_alignment_messages',
    'build_binary_alignment_prompt',
    'build_qwen_messages',
    'build_routing_prompt',
    'center_pad_image',
    'normalized_square_box_to_pixel_box',
    'original_pixel_box_to_normalized_square_box',
    'parse_binary_alignment_output',
    'parse_routing_output',
    'prepare_grounding_action_image',
    'render_candidate_box',
    'resize_crop_for_qwen',
]
