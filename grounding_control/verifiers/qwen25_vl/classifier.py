"""Standalone Qwen2.5-VL binary candidate-alignment classifier.

This module does not implement or import the four-way ActionVerifier contract.
It owns only the candidate-aware binary prompt, parsing, and audit metadata.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from ...models.qwen25_vl.runner import Qwen25VLRunner
from .inputs import CandidateVerificationInput
from .parser import parse_binary_alignment_output
from .prompt import (
    BINARY_IMAGE_MODES,
    BINARY_IMAGE_PROTOCOLS,
    build_binary_alignment_messages,
    build_binary_alignment_prompt,
)
from .rendering import (
    COORDINATE_SYSTEM,
    DEFAULT_QWEN_CROP_MIN_SIDE,
    render_candidate_box,
    resize_crop_for_qwen,
)


@dataclass(frozen=True)
class BinaryAlignmentLookup:
    aligned: Optional[bool]
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]


class Qwen25VLBinaryAlignmentClassifier:
    """Run only the binary object--candidate alignment protocol."""

    def __init__(
            self,
            runner: Qwen25VLRunner,
            *,
            crop_min_side: int = DEFAULT_QWEN_CROP_MIN_SIDE,
            parse_fail_open: bool = True):
        if runner is None:
            raise ValueError('runner is required')
        if (
                not isinstance(crop_min_side, int)
                or isinstance(crop_min_side, bool)
                or crop_min_side <= 28):
            raise ValueError('crop_min_side must be an integer greater than 28')
        self.runner = runner
        self.crop_min_side = int(crop_min_side)
        self.parse_fail_open = bool(parse_fail_open)

    def classify(
            self,
            candidate: CandidateVerificationInput,
            image_mode: str = 'crop_only') -> BinaryAlignmentLookup:
        if image_mode not in BINARY_IMAGE_MODES:
            raise ValueError(
                f'image_mode must be one of {BINARY_IMAGE_MODES}, '
                f'got {image_mode!r}'
            )
        if not isinstance(candidate.image, Image.Image):
            raise ValueError('CandidateVerificationInput.image must be a PIL image')
        if candidate.coordinate_system != COORDINATE_SYSTEM:
            raise ValueError(
                'Qwen verifier requires candidate_bbox in '
                f'{COORDINATE_SYSTEM!r}, got {candidate.coordinate_system!r}'
            )

        rendered = render_candidate_box(candidate.image, candidate.candidate_bbox)
        protocol = BINARY_IMAGE_PROTOCOLS[image_mode]
        prompt = build_binary_alignment_prompt(
            candidate.object_reference,
            image_mode=image_mode,
        )
        model_crop = (
            resize_crop_for_qwen(rendered.crop_image, self.crop_min_side)
            if protocol['uses_crop']
            else None
        )
        messages = build_binary_alignment_messages(
            crop_image=model_crop,
            prompt=prompt,
            annotated_image=(
                rendered.annotated_image
                if protocol['uses_bbox_image']
                else None
            ),
            image_mode=image_mode,
        )
        raw_response = self.runner.generate(messages)
        metadata = {
            'backend': f'qwen25_vl_binary_alignment_{image_mode}',
            'binary_image_mode': image_mode,
            'model_image_count': protocol['model_image_count'],
            'sample_id': candidate.sample_id,
            'raw_response': raw_response,
            'prompt': prompt,
            'coordinate_system': COORDINATE_SYSTEM,
            'original_image_size': list(rendered.original_size),
            'padded_square_size': rendered.padded_size,
            'padding_offset': list(rendered.padding_offset),
            'candidate_pixel_bbox_xyxy': list(rendered.pixel_bbox_xyxy),
            'candidate_crop_size': list(rendered.crop_image.size),
            'model_crop_size': (
                list(model_crop.size) if model_crop is not None else None
            ),
            'crop_min_side': self.crop_min_side,
            'runner_min_pixels': getattr(self.runner, 'min_pixels', None),
            'runner_max_pixels': getattr(self.runner, 'max_pixels', None),
        }
        try:
            parsed = parse_binary_alignment_output(raw_response)
        except (TypeError, ValueError) as error:
            if not self.parse_fail_open:
                raise
            return BinaryAlignmentLookup(
                aligned=None,
                confidence=None,
                error=f'{type(error).__name__}: {error}',
                metadata={**metadata, 'parse_failed': True},
            )
        return BinaryAlignmentLookup(
            aligned=parsed.aligned,
            confidence=parsed.confidence,
            error=None,
            metadata={
                **metadata,
                'parse_failed': False,
                'parsed_payload': parsed.payload,
            },
        )


__all__ = [
    'BinaryAlignmentLookup',
    'Qwen25VLBinaryAlignmentClassifier',
]
