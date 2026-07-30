"""Five-way Qwen2.5-VL implementation of the verifier backend contract."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from PIL import Image

from ...backend import VerificationRequest, VerifierBackend
from ...types import Box, VerificationLookup, VerificationResult
from .parser import (
    parse_binary_alignment_output,
    parse_routing_output,
    parse_verifier_output,
)
from .prompt import (
    BINARY_IMAGE_PROTOCOLS,
    BINARY_IMAGE_MODES,
    ROUTING_SYSTEM_PROMPT,
    build_binary_alignment_messages,
    build_binary_alignment_prompt,
    build_qwen_messages,
    build_routing_prompt,
    build_verification_prompt,
)
from .rendering import COORDINATE_SYSTEM, render_candidate_box
from .rendering import DEFAULT_QWEN_CROP_MIN_SIDE, resize_crop_for_qwen
from .runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
    Qwen25VLRunner,
)


@dataclass(frozen=True)
class CandidateVerificationInput:
    """Model-visible input shared by online and offline Qwen verification.

    It deliberately contains no target box, expected label, generated token
    span, or benchmark construction metadata.  Offline adapters therefore
    cannot accidentally leak supervision into the verifier prompt.
    """

    image: Image.Image
    object_reference: str
    candidate_bbox: Box
    sample_id: str = ''
    coordinate_system: str = COORDINATE_SYSTEM


@dataclass(frozen=True)
class BinaryAlignmentLookup:
    aligned: Optional[bool]
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RoutingClassificationLookup:
    status: Optional[str]
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any]


class Qwen25VLVerifierBackend(VerifierBackend):
    """Judge a candidate region with zero-shot five-way Qwen prompting."""

    def __init__(
            self,
            runner: Optional[Qwen25VLRunner] = None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            max_new_tokens: int = 64,
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: int = DEFAULT_MAX_PIXELS,
            crop_min_side: int = DEFAULT_QWEN_CROP_MIN_SIDE,
            parse_fail_open: bool = True):
        if runner is None:
            if not model_path:
                raise ValueError('model_path is required when runner is omitted')
            runner = LocalQwen25VLRunner(
                model_path=model_path,
                device=device,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        self.runner = runner
        if (
            not isinstance(crop_min_side, int)
            or isinstance(crop_min_side, bool)
            or crop_min_side <= 28
        ):
            raise ValueError('crop_min_side must be an integer greater than 28')
        self.crop_min_side = int(crop_min_side)
        self.parse_fail_open = bool(parse_fail_open)

    @staticmethod
    def _source_image(request: VerificationRequest) -> Image.Image:
        image = request.sample_context.get('image')
        if not isinstance(image, Image.Image):
            raise ValueError(
                'VerificationRequest.sample_context["image"] must be a PIL image'
            )
        return image

    def verify_candidate(
            self,
            candidate: CandidateVerificationInput,
            image_mode: str = 'marked_plus_crop',
    ) -> VerificationLookup:
        """Verify one model-visible candidate without online token metadata."""

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
        rendered = render_candidate_box(
            candidate.image,
            candidate.candidate_bbox,
        )
        protocol = BINARY_IMAGE_PROTOCOLS[image_mode]
        prompt = build_verification_prompt(
            object_reference=candidate.object_reference,
            candidate_bbox=candidate.candidate_bbox,
            image_mode=image_mode,
        )
        model_crop = (
            resize_crop_for_qwen(
                rendered.crop_image,
                self.crop_min_side,
            )
            if protocol['uses_crop']
            else None
        )
        messages = build_qwen_messages(
            annotated_image=(
                rendered.annotated_image
                if protocol['uses_bbox_image']
                else None
            ),
            crop_image=model_crop,
            prompt=prompt,
            image_mode=image_mode,
        )
        raw_response = self.runner.generate(messages)
        base_metadata = {
            'backend': f'qwen25_vl_zero_shot_{image_mode}',
            'image_mode': image_mode,
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
                list(model_crop.size)
                if model_crop is not None
                else None
            ),
            'crop_min_side': self.crop_min_side,
            'runner_min_pixels': getattr(self.runner, 'min_pixels', None),
            'runner_max_pixels': getattr(self.runner, 'max_pixels', None),
        }
        try:
            parsed = parse_verifier_output(raw_response)
        except (TypeError, ValueError) as error:
            if not self.parse_fail_open:
                raise
            return VerificationLookup(
                result=VerificationResult.uncertain(),
                error=f'{type(error).__name__}: {error}',
                metadata={
                    **base_metadata,
                    'status': None,
                    'parse_failed': True,
                },
            )
        return VerificationLookup(
            result=parsed.result,
            metadata={
                **base_metadata,
                'status': parsed.status,
                'parse_failed': False,
                'parsed_payload': parsed.payload,
            },
        )

    def verify(self, request: VerificationRequest) -> VerificationLookup:
        """Adapt the online controller request to the shared candidate input."""

        coordinate_system = request.sample_context.get(
            'coordinate_system',
            COORDINATE_SYSTEM,
        )
        image_mode = request.sample_context.get(
            'verifier_image_mode',
            'marked_plus_crop',
        )
        return self.verify_candidate(
            CandidateVerificationInput(
                image=self._source_image(request),
                object_reference=request.object_reference,
                candidate_bbox=request.candidate_bbox,
                sample_id=request.sample_id,
                coordinate_system=str(coordinate_system),
            ),
            image_mode=str(image_mode),
        )

    def classify_routing_candidate(
            self,
            candidate: CandidateVerificationInput,
            image_mode: str = 'bbox_image_only',
    ) -> RoutingClassificationLookup:
        """Choose no_action/relocate/expand/tighten for one candidate."""

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
        rendered = render_candidate_box(
            candidate.image,
            candidate.candidate_bbox,
        )
        protocol = BINARY_IMAGE_PROTOCOLS[image_mode]
        prompt = build_routing_prompt(
            object_reference=candidate.object_reference,
            candidate_bbox=candidate.candidate_bbox,
            image_mode=image_mode,
        )
        model_crop = (
            resize_crop_for_qwen(
                rendered.crop_image,
                self.crop_min_side,
            )
            if protocol['uses_crop']
            else None
        )
        messages = build_qwen_messages(
            annotated_image=(
                rendered.annotated_image
                if protocol['uses_bbox_image']
                else None
            ),
            crop_image=model_crop,
            prompt=prompt,
            image_mode=image_mode,
            system_prompt=ROUTING_SYSTEM_PROMPT,
        )
        raw_response = self.runner.generate(messages)
        metadata = {
            'backend': f'qwen25_vl_routing_four_way_{image_mode}',
            'image_mode': image_mode,
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
                list(model_crop.size)
                if model_crop is not None
                else None
            ),
            'crop_min_side': self.crop_min_side,
            'runner_min_pixels': getattr(self.runner, 'min_pixels', None),
            'runner_max_pixels': getattr(self.runner, 'max_pixels', None),
        }
        try:
            parsed = parse_routing_output(raw_response)
        except (TypeError, ValueError) as error:
            if not self.parse_fail_open:
                raise
            return RoutingClassificationLookup(
                status=None,
                confidence=None,
                error=f'{type(error).__name__}: {error}',
                metadata={**metadata, 'parse_failed': True},
            )
        return RoutingClassificationLookup(
            status=parsed.status,
            confidence=parsed.confidence,
            error=None,
            metadata={
                **metadata,
                'parse_failed': False,
                'parsed_payload': parsed.payload,
            },
        )

    def verify_binary_alignment_candidate(
            self,
            candidate: CandidateVerificationInput,
            image_mode: str = 'crop_only',
    ) -> BinaryAlignmentLookup:
        """Run binary alignment with crop-only or marked-scene context."""

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
        rendered = render_candidate_box(
            candidate.image,
            candidate.candidate_bbox,
        )
        protocol = BINARY_IMAGE_PROTOCOLS[image_mode]
        prompt = build_binary_alignment_prompt(
            candidate.object_reference,
            image_mode=image_mode,
        )
        model_crop = (
            resize_crop_for_qwen(
                rendered.crop_image,
                self.crop_min_side,
            )
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
                list(model_crop.size)
                if model_crop is not None
                else None
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
