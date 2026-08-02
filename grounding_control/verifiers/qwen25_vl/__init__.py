"""Qwen2.5-VL verifier implementations.

Reusable Qwen model execution and object-to-box prediction live under
``grounding_control.models.qwen25_vl``.  This package exports only the binary
alignment mainline; retained action classification lives under
``grounding_control.four_way.verifiers.qwen25_vl``.
"""

from .classifier import (
    BinaryAlignmentLookup,
    Qwen25VLBinaryAlignmentClassifier,
)
from .backend import (
    QWEN_ALIGNMENT_SCORE_SEMANTICS,
    Qwen25VLAlignmentVerifierBackend,
    binary_lookup_to_alignment_output,
)
from .inputs import CandidateVerificationInput
from .parser import (
    ParsedBinaryAlignmentOutput,
    parse_binary_alignment_output,
)
from .prompt import (
    BINARY_IMAGE_MODES,
    BINARY_IMAGE_PROTOCOLS,
    build_binary_alignment_messages,
    build_binary_alignment_prompt,
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
    'ParsedBinaryAlignmentOutput',
    'Qwen25VLAlignmentVerifierBackend',
    'Qwen25VLBinaryAlignmentClassifier',
    'QWEN_ALIGNMENT_SCORE_SEMANTICS',
    'RenderedCandidate',
    'build_binary_alignment_messages',
    'build_binary_alignment_prompt',
    'binary_lookup_to_alignment_output',
    'center_pad_image',
    'normalized_square_box_to_pixel_box',
    'original_pixel_box_to_normalized_square_box',
    'parse_binary_alignment_output',
    'render_candidate_box',
    'resize_crop_for_qwen',
]
