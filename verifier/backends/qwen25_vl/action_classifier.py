"""Four-way grounding-action classifier backed by option likelihood."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .action_prompt import (
    GROUNDING_ACTION_IMAGE_MODES,
    GROUNDING_ACTION_OPTIONS,
    build_grounding_action_messages,
    build_grounding_action_prompt,
)
from .option_likelihood import (
    SingleTokenOptionLikelihoodRunner,
    decide_from_option_scores,
)
from .runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
)


PixelBox = Tuple[float, float, float, float]
IntegerPixelBox = Tuple[int, int, int, int]
QWEN_IMAGE_FACTOR = 28
QWEN_MAX_ASPECT_RATIO = 200
COORDINATE_ROUNDING_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GroundingActionInput:
    """Leakage-safe clean image, reference, and original-image pixel bbox."""

    image: Image.Image
    object_reference: str
    candidate_bbox_pixel_xyxy: PixelBox
    sample_id: str = ''


@dataclass(frozen=True)
class GroundingActionLookup:
    status: str
    confidence: float
    error: Optional[str]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class PreparedGroundingActionImage:
    clean_image: Image.Image
    marked_image: Image.Image
    original_size: Tuple[int, int]
    model_size: Tuple[int, int]
    candidate_bbox_model_xyxy: IntegerPixelBox


def qwen_smart_resize_size(
        image_size: Sequence[int],
        min_pixels: int,
        max_pixels: int,
        factor: int = QWEN_IMAGE_FACTOR,
) -> Tuple[int, int]:
    """Reproduce Qwen2.5-VL smart-resize dimensions as ``(width, height)``."""

    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError('image_size must be (width, height)')
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError('image width and height must be positive')
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError('invalid Qwen min/max pixel bounds')
    if factor <= 0:
        raise ValueError('resize factor must be positive')
    if max(width, height) / min(width, height) > QWEN_MAX_ASPECT_RATIO:
        raise ValueError(
            f'image aspect ratio must be at most {QWEN_MAX_ASPECT_RATIO}'
        )

    resized_width = max(factor, round(width / factor) * factor)
    resized_height = max(factor, round(height / factor) * factor)
    if resized_width * resized_height > max_pixels:
        beta = math.sqrt((width * height) / max_pixels)
        resized_width = math.floor((width / beta) / factor) * factor
        resized_height = math.floor((height / beta) / factor) * factor
    elif resized_width * resized_height < min_pixels:
        beta = math.sqrt(min_pixels / (width * height))
        resized_width = math.ceil((width * beta) / factor) * factor
        resized_height = math.ceil((height * beta) / factor) * factor
    if resized_width <= 0 or resized_height <= 0:
        raise ValueError('Qwen smart resize produced an empty image')
    return resized_width, resized_height


def _validate_original_pixel_box(
        values: Sequence[float],
        image_size: Tuple[int, int],
) -> PixelBox:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('candidate pixel bbox must have four coordinates')
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f'candidate pixel bbox is non-finite: {box}')
    width, height = image_size
    if not (
        0 <= box[0] < box[2] <= width
        and 0 <= box[1] < box[3] <= height
    ):
        raise ValueError(
            f'candidate pixel bbox {box} is outside {width}x{height}'
        )
    return box  # type: ignore[return-value]


def prepare_grounding_action_image(
        image: Image.Image,
        candidate_bbox_pixel_xyxy: Sequence[float],
        min_pixels: int,
        max_pixels: int,
        box_color: Tuple[int, int, int] = (255, 0, 0),
        line_width: int = 4,
) -> PreparedGroundingActionImage:
    """Resize a clean source image and the bbox into one exact Qwen frame."""

    if not isinstance(image, Image.Image):
        raise TypeError('image must be a PIL.Image.Image')
    if not isinstance(line_width, int) or isinstance(line_width, bool):
        raise TypeError('line_width must be an integer')
    if line_width <= 0:
        raise ValueError('line_width must be positive')
    source = image.convert('RGB')
    original_size = source.size
    box = _validate_original_pixel_box(
        candidate_bbox_pixel_xyxy,
        original_size,
    )
    model_size = qwen_smart_resize_size(
        original_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    clean = source.resize(model_size, Image.Resampling.BICUBIC)
    scale_x = model_size[0] / original_size[0]
    scale_y = model_size[1] / original_size[1]
    x_min = max(0, min(
        model_size[0] - 1,
        int(math.floor(box[0] * scale_x)),
    ))
    y_min = max(0, min(
        model_size[1] - 1,
        int(math.floor(box[1] * scale_y)),
    ))
    x_max = max(x_min + 1, min(
        model_size[0],
        int(math.ceil(
            box[2] * scale_x - COORDINATE_ROUNDING_TOLERANCE
        )),
    ))
    y_max = max(y_min + 1, min(
        model_size[1],
        int(math.ceil(
            box[3] * scale_y - COORDINATE_ROUNDING_TOLERANCE
        )),
    ))
    model_box = (x_min, y_min, x_max, y_max)
    marked = clean.copy()
    ImageDraw.Draw(marked).rectangle(
        (x_min, y_min, x_max - 1, y_max - 1),
        outline=box_color,
        width=line_width,
    )
    return PreparedGroundingActionImage(
        clean_image=clean,
        marked_image=marked,
        original_size=original_size,
        model_size=model_size,
        candidate_bbox_model_xyxy=model_box,
    )


class Qwen25VLGroundingActionClassifier:
    """Classify A/B/C/D from one multimodal next-token distribution."""

    def __init__(
            self,
            runner: Optional[SingleTokenOptionLikelihoodRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: int = DEFAULT_MAX_PIXELS,
            attn_implementation: str = 'sdpa'):
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
        self.runner = runner
        self.min_pixels = int(getattr(runner, 'min_pixels', min_pixels))
        self.max_pixels = int(getattr(runner, 'max_pixels', max_pixels))

    def classify(
            self,
            candidate: GroundingActionInput,
            image_mode: str = 'raw_image',
    ) -> GroundingActionLookup:
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
        prompt = build_grounding_action_prompt(
            object_reference=candidate.object_reference,
            candidate_bbox_xyxy=prepared.candidate_bbox_model_xyxy,
            image_size=prepared.model_size,
            image_mode=image_mode,
        )
        messages = build_grounding_action_messages(model_image, prompt)
        scores = self.runner.score_single_token_options(
            messages,
            GROUNDING_ACTION_OPTIONS,
        )
        decision = decide_from_option_scores(scores)
        return GroundingActionLookup(
            status=decision.label,
            confidence=decision.confidence,
            error=None,
            metadata={
                'backend': (
                    'qwen25_vl_grounding_action_option_likelihood_'
                    f'{image_mode}'
                ),
                'image_mode': image_mode,
                'model_image_count': 1,
                'sample_id': candidate.sample_id,
                'prompt': prompt,
                'options': dict(GROUNDING_ACTION_OPTIONS),
                'option_negative_log_likelihoods': (
                    decision.negative_log_likelihoods
                ),
                'option_normalized_probabilities': (
                    decision.normalized_probabilities
                ),
                'option_token_ids': decision.token_ids,
                'option_nll_margin': decision.nll_margin,
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
            },
        )
