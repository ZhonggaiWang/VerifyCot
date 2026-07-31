"""Qwen2.5-VL zero-shot object--coordinate verifier."""

from .action_classifier import (
    GroundingActionInput,
    GroundingActionLookup,
    PreparedGroundingActionImage,
    Qwen25VLGroundingActionClassifier,
    prepare_grounding_action_image,
    qwen_smart_resize_size,
)
from .action_prompt import (
    GROUNDING_ACTION_IMAGE_MODES,
    GROUNDING_ACTION_OPTIONS,
    GROUNDING_ACTION_SYSTEM_PROMPT,
    build_grounding_action_messages,
    build_grounding_action_prompt,
)
from .backend import (
    BinaryAlignmentLookup,
    CandidateVerificationInput,
    Qwen25VLVerifierBackend,
    RoutingClassificationLookup,
)
from .grounding_classifier import (
    GroundingGeometryLookup,
    Qwen25VLGroundingGeometryClassifier,
)
from .grounding_geometry import (
    GroundingGeometryDecision,
    route_from_grounding_geometry,
)
from .grounding_parser import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    ParsedReferenceGroundingBox,
    parse_reference_grounding_box,
    parse_reference_grounding_box_details,
)
from .grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
    GROUNDING_SYSTEM_PROMPT,
    GROUNDING_SYSTEM_PROMPTS,
    build_reference_grounding_messages,
    build_reference_grounding_prompt,
)
from .option_likelihood import (
    OptionLikelihoodDecision,
    SingleTokenOptionLikelihoodRunner,
    SingleTokenOptionScores,
    decide_from_option_scores,
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
    'build_grounding_action_messages',
    'build_grounding_action_prompt',
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
    'DEFAULT_BOUNDARY_TOLERANCE_PIXELS',
    'DEFAULT_GROUNDING_PROMPT_PROTOCOL',
    'DEFAULT_QWEN_CROP_MIN_SIDE',
    'decide_from_option_scores',
    'GROUNDING_ACTION_IMAGE_MODES',
    'GROUNDING_ACTION_OPTIONS',
    'GROUNDING_ACTION_SYSTEM_PROMPT',
    'GroundingActionInput',
    'GroundingActionLookup',
    'GroundingGeometryDecision',
    'GroundingGeometryLookup',
    'GROUNDING_SYSTEM_PROMPT',
    'GROUNDING_PROMPT_PROTOCOLS',
    'GROUNDING_SYSTEM_PROMPTS',
    'LocalQwen25VLRunner',
    'normalized_square_box_to_pixel_box',
    'original_pixel_box_to_normalized_square_box',
    'resize_crop_for_qwen',
    'ParsedVerifierOutput',
    'ParsedBinaryAlignmentOutput',
    'ParsedRoutingOutput',
    'ParsedReferenceGroundingBox',
    'parse_binary_alignment_output',
    'parse_routing_output',
    'parse_verifier_output',
    'Qwen25VLRunner',
    'Qwen25VLGroundingActionClassifier',
    'Qwen25VLGroundingGeometryClassifier',
    'Qwen25VLVerifierBackend',
    'RoutingClassificationLookup',
    'ROUTING_STATUSES',
    'render_candidate_box',
    'RenderedCandidate',
    'OptionLikelihoodDecision',
    'prepare_grounding_action_image',
    'PreparedGroundingActionImage',
    'qwen_smart_resize_size',
    'parse_reference_grounding_box',
    'parse_reference_grounding_box_details',
    'route_from_grounding_geometry',
    'SingleTokenOptionLikelihoodRunner',
    'SingleTokenOptionScores',
    'STATUSES',
    'build_reference_grounding_messages',
    'build_reference_grounding_prompt',
    'build_routing_prompt',
]
