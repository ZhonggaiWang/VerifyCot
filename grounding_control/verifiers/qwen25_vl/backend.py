"""Binary alignment adapter for the existing Qwen2.5-VL verifier.

The underlying prompt returns a hard ``aligned`` label and confidence in that
label.  The pre-commit binary controller instead needs a *signed* score whose
direction is fixed: larger always means that the candidate is more likely to
be aligned.  This adapter performs that conversion explicitly and records
that the model's self-reported confidence is not calibrated.
"""

from typing import Any, Dict, Optional, Protocol

from ...contracts.alignment_verifier import (
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
)
from ...contracts.requests import CandidateAlignmentRequest
from ...models.qwen25_vl.runner import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    LocalQwen25VLRunner,
)
from .classifier import (
    BinaryAlignmentLookup,
    Qwen25VLBinaryAlignmentClassifier,
)
from .inputs import CandidateVerificationInput
from .prompt import BINARY_IMAGE_MODES
from .rendering import COORDINATE_SYSTEM, DEFAULT_QWEN_CROP_MIN_SIDE


QWEN_ALIGNMENT_SCORE_SEMANTICS = (
    'qwen_self_reported_label_confidence_transformed_uncalibrated'
)


class _BinaryClassifierProvider(Protocol):
    """Shape of the retained legacy backend accepted for compatibility."""

    binary_classifier: Qwen25VLBinaryAlignmentClassifier


def binary_lookup_to_alignment_output(
        lookup: BinaryAlignmentLookup,
        *,
        extra_metadata: Optional[Dict[str, Any]] = None,
) -> AlignmentVerifierOutput:
    """Convert label confidence into a directionally stable alignment score."""

    metadata = {
        **dict(lookup.metadata),
        **dict(extra_metadata or {}),
        'binary_aligned_label': lookup.aligned,
        'binary_label_confidence': lookup.confidence,
        'binary_label_confidence_semantics': (
            'self_reported_confidence_in_the_predicted_binary_label'
        ),
        'alignment_score_transformation': (
            'confidence_if_aligned_else_one_minus_confidence'
        ),
        'alignment_score_calibrated': False,
    }
    if lookup.aligned is None or lookup.confidence is None:
        return AlignmentVerifierOutput.unknown(
            error=lookup.error or 'binary_alignment_lookup_unavailable',
            score_kind='self_reported_probability',
            score_semantics=QWEN_ALIGNMENT_SCORE_SEMANTICS,
            metadata=metadata,
        )

    confidence = float(lookup.confidence)
    if not 0.5 <= confidence <= 1.0:
        return AlignmentVerifierOutput.unknown(
            error=(
                'binary label confidence must be in [0.5, 1.0] when it '
                'measures confidence in the emitted label'
            ),
            score_semantics=QWEN_ALIGNMENT_SCORE_SEMANTICS,
            score_kind='self_reported_probability',
            metadata={
                **metadata,
                'binary_label_confidence_inconsistent': True,
            },
        )
    score = confidence if lookup.aligned else 1.0 - confidence
    return AlignmentVerifierOutput(
        alignment_score=score,
        score_kind='self_reported_probability',
        score_semantics=QWEN_ALIGNMENT_SCORE_SEMANTICS,
        abstained=False,
        error=None,
        metadata=metadata,
    )


class Qwen25VLAlignmentVerifierBackend(AlignmentVerifierBackend):
    """Expose Qwen's candidate-aware binary prompt through the new contract."""

    def __init__(
            self,
            backend: Optional[_BinaryClassifierProvider] = None,
            *,
            classifier: Optional[Qwen25VLBinaryAlignmentClassifier] = None,
            image_mode: str = 'bbox_image_only',
            runner=None,
            model_path: Optional[str] = None,
            device: str = 'cuda:0',
            dtype: str = 'bfloat16',
            max_new_tokens: int = 64,
            min_pixels: int = DEFAULT_MIN_PIXELS,
            max_pixels: int = DEFAULT_MAX_PIXELS,
            crop_min_side: int = DEFAULT_QWEN_CROP_MIN_SIDE,
            parse_fail_open: bool = True):
        if image_mode not in BINARY_IMAGE_MODES:
            raise ValueError(
                f'image_mode must be one of {BINARY_IMAGE_MODES}, '
                f'got {image_mode!r}'
            )
        configured_sources = sum(
            value is not None for value in (backend, classifier, runner)
        ) + int(model_path is not None)
        if configured_sources > 1:
            raise ValueError(
                'configure exactly one of backend, classifier, runner, or '
                'model_path'
            )
        if backend is not None:
            classifier = backend.binary_classifier
        elif classifier is None:
            if runner is None:
                if not model_path:
                    raise ValueError(
                        'model_path is required when runner/classifier is omitted'
                    )
                runner = LocalQwen25VLRunner(
                    model_path=model_path,
                    device=device,
                    dtype=dtype,
                    max_new_tokens=max_new_tokens,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
            classifier = Qwen25VLBinaryAlignmentClassifier(
                runner,
                crop_min_side=crop_min_side,
                parse_fail_open=parse_fail_open,
            )
        self.classifier = classifier
        self.image_mode = str(image_mode)

    def verify_alignment(
            self,
            request: CandidateAlignmentRequest,
    ) -> AlignmentVerifierOutput:
        if not isinstance(request, CandidateAlignmentRequest):
            raise TypeError(
                'Qwen alignment verifier requires CandidateAlignmentRequest'
            )
        image_mode = str(request.image_mode or self.image_mode)
        coordinate_system = str(request.coordinate_system or COORDINATE_SYSTEM)
        lookup = self.classifier.classify(
            CandidateVerificationInput(
                image=request.visual.image,
                object_reference=request.object_reference,
                candidate_bbox=request.candidate_bbox,
                sample_id=request.sample_id,
                coordinate_system=coordinate_system,
            ),
            image_mode=image_mode,
        )
        return binary_lookup_to_alignment_output(
            lookup,
            extra_metadata={
                'alignment_backend': 'qwen25_vl_binary_alignment',
                'requested_image_mode': image_mode,
            },
        )


__all__ = [
    'QWEN_ALIGNMENT_SCORE_SEMANTICS',
    'Qwen25VLAlignmentVerifierBackend',
    'binary_lookup_to_alignment_output',
]
