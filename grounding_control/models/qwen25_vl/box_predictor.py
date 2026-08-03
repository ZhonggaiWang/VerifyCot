"""Adapt Qwen2.5-VL coordinate generation to the box-predictor capability."""

from typing import Any, Dict, Optional

from ..box_predictor import BoxPrediction, BoxPredictionRequest
from .grounding_parser import (
    DEFAULT_BOUNDARY_TOLERANCE_PIXELS,
    parse_reference_grounding_box_details,
)
from .grounding_prompt import (
    DEFAULT_GROUNDING_PROMPT_PROTOCOL,
    build_reference_grounding_messages,
    build_reference_grounding_prompt,
)
from .runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    Qwen25VLRunner,
)
from .preprocessing import prepare_reference_image


class Qwen25VLBoxPredictor:
    """Generate one reference box and return it on the original image."""

    def __init__(
            self,
            runner: Qwen25VLRunner,
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: Optional[int] = DEFAULT_MAX_PIXELS,
            boundary_tolerance_pixels: float = (
                DEFAULT_BOUNDARY_TOLERANCE_PIXELS
            ),
            prompt_protocol: str = DEFAULT_GROUNDING_PROMPT_PROTOCOL):
        self.runner = runner
        self.min_pixels = int(getattr(runner, 'min_pixels', min_pixels))
        runner_max_pixels = getattr(runner, 'max_pixels', max_pixels)
        self.max_pixels = (
            None if runner_max_pixels is None else int(runner_max_pixels)
        )
        self.boundary_tolerance_pixels = float(boundary_tolerance_pixels)
        self.prompt_protocol = prompt_protocol

    def predict(self, request: BoxPredictionRequest) -> BoxPrediction:
        width, height = request.image.size
        prepared = prepare_reference_image(
            request.image,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        prompt = build_reference_grounding_prompt(
            request.object_reference,
            prepared.model_size,
            'raw_image',
            prompt_protocol=self.prompt_protocol,
        )
        messages = build_reference_grounding_messages(
            prepared.image,
            prompt,
            prompt_protocol=self.prompt_protocol,
        )
        raw_response = self.runner.generate(messages)
        metadata: Dict[str, Any] = {
            'backend': 'qwen25_vl_box_predictor',
            'sample_id': request.sample_id,
            'prompt': prompt,
            'prompt_protocol': self.prompt_protocol,
            'raw_response': raw_response,
            'original_image_size': list(prepared.original_size),
            'model_image_size': list(prepared.model_size),
            'configured_max_pixels': self.max_pixels,
            'effective_model_pixels': (
                prepared.model_size[0] * prepared.model_size[1]
            ),
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'parse_failed': False,
        }
        try:
            parsed = parse_reference_grounding_box_details(
                raw_response,
                prepared.model_size,
                boundary_tolerance_pixels=self.boundary_tolerance_pixels,
            )
        except ValueError as error:
            metadata['parse_failed'] = True
            return BoxPrediction(
                bbox_pixel_xyxy=None,
                confidence=None,
                error=f'{type(error).__name__}: {error}',
                metadata=metadata,
            )

        model_width, model_height = prepared.model_size
        scale_x = width / float(model_width)
        scale_y = height / float(model_height)
        original_box = (
            parsed.box[0] * scale_x,
            parsed.box[1] * scale_y,
            parsed.box[2] * scale_x,
            parsed.box[3] * scale_y,
        )
        metadata.update({
            'predicted_box_model_pixel_xyxy': list(parsed.box),
            'predicted_box_original_pixel_xyxy': list(original_box),
            'boundary_clipped': parsed.boundary_clipped,
            'boundary_clipped_sides': list(parsed.clipped_sides),
        })
        return BoxPrediction(
            bbox_pixel_xyxy=original_box,
            confidence=None,
            error=None,
            metadata=metadata,
        )
