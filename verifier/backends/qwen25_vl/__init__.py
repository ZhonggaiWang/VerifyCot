"""Qwen2.5-VL zero-shot object--coordinate verifier."""

from .backend import (
    BinaryAlignmentLookup,
    CandidateVerificationInput,
    Qwen25VLVerifierBackend,
    RoutingClassificationLookup,
)
from .parser import (
    ParsedBinaryAlignmentOutput,
    ParsedRoutingOutput,
    ParsedVerifierOutput,
    parse_binary_alignment_output,
    parse_routing_output,
    parse_verifier_output,
)
from .prompt import (
    BINARY_IMAGE_PROTOCOLS,
    BINARY_IMAGE_MODES,
    ROUTING_STATUSES,
    STATUSES,
    build_binary_alignment_messages,
    build_binary_alignment_prompt,
    build_qwen_messages,
    build_routing_prompt,
    build_verification_prompt,
)
from .rendering import (
    COORDINATE_SYSTEM,
    DEFAULT_QWEN_CROP_MIN_SIDE,
    RenderedCandidate,
    center_pad_image,
    normalized_square_box_to_pixel_box,
    original_pixel_box_to_normalized_square_box,
    resize_crop_for_qwen,
    render_candidate_box,
)
from .runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLRunner,
)

__all__ = [
    'build_qwen_messages',
    'build_verification_prompt',
    'build_binary_alignment_messages',
    'build_binary_alignment_prompt',
    'BINARY_IMAGE_PROTOCOLS',
    'BINARY_IMAGE_MODES',
    'BinaryAlignmentLookup',
    'CandidateVerificationInput',
    'center_pad_image',
    'COORDINATE_SYSTEM',
    'DEFAULT_MAX_PIXELS',
    'DEFAULT_MIN_PIXELS',
    'DEFAULT_QWEN_CROP_MIN_SIDE',
    'LocalQwen25VLRunner',
    'normalized_square_box_to_pixel_box',
    'original_pixel_box_to_normalized_square_box',
    'resize_crop_for_qwen',
    'ParsedVerifierOutput',
    'ParsedBinaryAlignmentOutput',
    'ParsedRoutingOutput',
    'parse_binary_alignment_output',
    'parse_routing_output',
    'parse_verifier_output',
    'Qwen25VLRunner',
    'Qwen25VLVerifierBackend',
    'RoutingClassificationLookup',
    'ROUTING_STATUSES',
    'render_candidate_box',
    'RenderedCandidate',
    'STATUSES',
    'build_routing_prompt',
]
