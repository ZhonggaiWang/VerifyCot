"""Generate a reference box, then route the candidate by box geometry."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional

from .action_classifier import (
    GroundingActionInput,
    prepare_grounding_action_image,
)
from .action_prompt import GROUNDING_ACTION_IMAGE_MODES
from .grounding_geometry import route_from_grounding_geometry
from .grounding_parser import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    parse_reference_grounding_box_details,
)
from .grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    GROUNDING_PROMPT_PROTOCOLS,
    build_reference_grounding_messages,
    build_reference_grounding_prompt,
)
from .runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLRunner,
)


@dataclass(frozen=True)
class GroundingGeometryLookup:
    status: Optional[str]
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]


class Qwen25VLGroundingGeometryClassifier:
    """Use Qwen as a grounder and deterministic geometry as the classifier."""

    def __init__(
            self,
            runner: Optional[Qwen25VLRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: int = DEFAULT_MAX_PIXELS,
            attn_implementation: str = 'sdpa',
            accept_iou_threshold: float = 0.5,
            containment_threshold: float = 0.7,
            boundary_tolerance_pixels: float = (
                DEFAULT_BOUNDARY_TOLERANCE_PIXELS
            ),
            prompt_protocol: str = DEFAULT_GROUNDING_PROMPT_PROTOCOL):
        if runner is None:
            if not model_path:
                raise ValueError('model_path is required when runner is omitted')
            runner = LocalQwen25VLRunner(
                model_path=model_path,
                device=device,
                dtype=dtype,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                attn_implementation=attn_implementation,
            )
        if not 0.0 < accept_iou_threshold <= 1.0:
            raise ValueError('accept_iou_threshold must be in (0, 1]')
        if not 0.0 < containment_threshold <= 1.0:
            raise ValueError('containment_threshold must be in (0, 1]')
        if isinstance(boundary_tolerance_pixels, bool):
            raise TypeError('boundary_tolerance_pixels must be numeric')
        boundary_tolerance = float(boundary_tolerance_pixels)
        if not math.isfinite(boundary_tolerance) or boundary_tolerance < 0:
            raise ValueError(
                'boundary_tolerance_pixels must be finite and non-negative'
            )
        if prompt_protocol not in GROUNDING_PROMPT_PROTOCOLS:
            raise ValueError(
                f'prompt_protocol must be one of '
                f'{GROUNDING_PROMPT_PROTOCOLS}, got {prompt_protocol!r}'
            )
        self.runner = runner
        self.min_pixels = int(getattr(runner, 'min_pixels', min_pixels))
        self.max_pixels = int(getattr(runner, 'max_pixels', max_pixels))
        self.accept_iou_threshold = float(accept_iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.boundary_tolerance_pixels = boundary_tolerance
        self.prompt_protocol = prompt_protocol

    def classify(
            self,
            candidate: GroundingActionInput,
            image_mode: str = 'raw_image',
    ) -> GroundingGeometryLookup:
        if image_mode not in GROUNDING_ACTION_IMAGE_MODES:
            raise ValueError(
                f'image_mode must be one of {GROUNDING_ACTION_IMAGE_MODES}, '
                f'got {image_mode!r}'
            )
        prepared = prepare_grounding_action_image(
            candidate.image,
            candidate.candidate_bbox_pixel_xyxy,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        model_image = (
            prepared.clean_image
            if image_mode == 'raw_image'
            else prepared.marked_image
        )
        prompt = build_reference_grounding_prompt(
            candidate.object_reference,
            prepared.model_size,
            image_mode,
            prompt_protocol=self.prompt_protocol,
        )
        messages = build_reference_grounding_messages(
            model_image,
            prompt,
            prompt_protocol=self.prompt_protocol,
        )
        raw_response = self.runner.generate(messages)
        metadata: Dict[str, Any] = {
            'backend': (
                f'qwen25_vl_grounding_geometry_router_{image_mode}'
            ),
            'image_mode': image_mode,
            'model_image_count': 1,
            'sample_id': candidate.sample_id,
            'prompt': prompt,
            'prompt_protocol': self.prompt_protocol,
            'raw_response': raw_response,
            'coordinate_system': (
                'absolute_xyxy_on_qwen_smart_resized_image'
            ),
            'original_image_size': list(prepared.original_size),
            'model_image_size': list(prepared.model_size),
            'candidate_original_pixel_bbox_xyxy': list(
                candidate.candidate_bbox_pixel_xyxy
            ),
            'candidate_model_pixel_bbox_xyxy': list(
                prepared.candidate_bbox_model_xyxy
            ),
            'runner_min_pixels': self.min_pixels,
            'runner_max_pixels': self.max_pixels,
            'parse_failed': False,
            'grounding_box_raw_model_pixel_xyxy': None,
            'grounding_box_model_pixel_xyxy': None,
            'grounding_boundary_clipped': None,
            'grounding_boundary_clipped_sides': [],
            'grounding_boundary_tolerance_pixels': (
                self.boundary_tolerance_pixels
            ),
        }
        try:
            parsed_box = parse_reference_grounding_box_details(
                raw_response,
                prepared.model_size,
                boundary_tolerance_pixels=self.boundary_tolerance_pixels,
            )
        except ValueError as error:
            metadata['parse_failed'] = True
            return GroundingGeometryLookup(
                status=None,
                confidence=None,
                error=f'{type(error).__name__}: {error}',
                metadata=metadata,
            )

        grounding_box = parsed_box.box
        decision = route_from_grounding_geometry(
            prepared.candidate_bbox_model_xyxy,
            grounding_box,
            accept_iou_threshold=self.accept_iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        metadata['grounding_box_raw_model_pixel_xyxy'] = list(
            parsed_box.raw_box
        )
        metadata['grounding_box_model_pixel_xyxy'] = list(grounding_box)
        metadata['grounding_boundary_clipped'] = (
            parsed_box.boundary_clipped
        )
        metadata['grounding_boundary_clipped_sides'] = list(
            parsed_box.clipped_sides
        )
        metadata['geometry'] = asdict(decision)
        return GroundingGeometryLookup(
            status=decision.action,
            confidence=None,
            error=None,
            metadata=metadata,
        )
