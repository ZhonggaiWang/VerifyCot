"""Grounding DINO compositions for offline and online geometry verification."""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from ..contracts import (
    CandidateAlignmentRequest,
)
from ..contracts.alignment_verifier import (
    AlignmentVerifierBackend,
    AlignmentVerifierOutput,
)
from ..models.grounding_dino import (
    GroundingDinoBoxPredictor,
    GroundingDinoRunner,
)
from .box_geometry import (
    PaddedGeometryComparator,
    PaddedGeometryVerificationInput,
)


class GroundingDinoAlignmentScorer:
    """Threshold-free DINO localization comparison for binary verification.

    This classifier intentionally owns neither a four-way verifier nor action
    thresholds.  It predicts one grounding box, compares it with the VoCoT
    candidate in the padded coordinate frame, and exposes the raw IoU proxy.
    """

    def __init__(
            self,
            runner: GroundingDinoRunner,
            top_k_log: int = 20):
        self.runner = runner
        self.top_k_log = int(top_k_log)
        self.predictor = GroundingDinoBoxPredictor(
            runner,
            top_k_log=top_k_log,
        )
        self.padded_comparator = PaddedGeometryComparator(
            self.predictor,
            backend_name=(
                'grounding_dino_geometry_comparator_'
                'vocot_padded_normalized'
            ),
        )

    def classify_padded_alignment(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> AlignmentVerifierOutput:
        """Return candidate-versus-selected-DINO IoU as the signed score.

        The DINO detector score remains audit metadata.  It is confidence in
        the selected detection, not confidence that the VoCoT candidate and
        object reference are aligned, and is therefore never used as the
        alignment score.
        """

        comparison = self.padded_comparator.compare(candidate)
        metadata = {
            **dict(comparison.metadata),
            'alignment_backend': 'grounding_dino_selected_box_geometry',
            'detector_confidence': comparison.metadata.get(
                'selected_grounding_score'
            ),
            'detector_confidence_used_as_alignment_score': False,
            'alignment_score_calibrated': False,
            'geometry_measurement': (
                None
                if comparison.measurement is None
                else asdict(comparison.measurement)
            ),
        }
        if comparison.measurement is None:
            return AlignmentVerifierOutput.unknown(
                error=comparison.error or 'box_prediction_failed',
                score_kind='iou_proxy',
                score_semantics=DINO_ALIGNMENT_SCORE_SEMANTICS,
                metadata=metadata,
            )
        return AlignmentVerifierOutput(
            alignment_score=float(comparison.measurement.iou),
            score_kind='iou_proxy',
            score_semantics=DINO_ALIGNMENT_SCORE_SEMANTICS,
            abstained=False,
            error=comparison.error,
            metadata=metadata,
        )


class PaddedGeometryAlignmentScorer(Protocol):
    def classify_padded_alignment(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> AlignmentVerifierOutput:
        ...


def _alignment_request_image(
        request: CandidateAlignmentRequest) -> Image.Image:
    image: Any = request.visual.image
    if isinstance(image, Image.Image):
        return image.convert('RGB').copy()
    image_path = request.visual.image_path
    if isinstance(image_path, (str, Path)):
        path = Path(image_path)
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert('RGB').copy()
    raise ValueError(
        'DINO alignment verifier requires visual.image as a PIL image or '
        'visual.image_path as a readable path'
    )


DINO_ALIGNMENT_SCORE_SEMANTICS = (
    'candidate_selected_grounding_iou_proxy_uncalibrated'
)


class GroundingDinoAlignmentVerifierBackend(AlignmentVerifierBackend):
    """Expose DINO candidate-versus-top-1 geometry as a binary score."""

    def __init__(self, classifier: PaddedGeometryAlignmentScorer):
        self.classifier = classifier

    def verify_alignment(
            self,
            request: CandidateAlignmentRequest,
    ) -> AlignmentVerifierOutput:
        if not isinstance(request, CandidateAlignmentRequest):
            raise TypeError(
                'DINO alignment verifier requires CandidateAlignmentRequest'
            )
        return self.classifier.classify_padded_alignment(
            PaddedGeometryVerificationInput(
                image=_alignment_request_image(request),
                object_reference=request.object_reference,
                candidate_bbox_padded_normalized_xyxy=(
                    request.candidate_bbox
                ),
                sample_id=request.sample_id,
            )
        )


# Short-term compatibility aliases.  They intentionally reference the exact
# canonical class objects rather than defining subclasses or duplicate types.
GroundingDinoAlignmentClassifier = GroundingDinoAlignmentScorer
PaddedGeometryAlignmentClassifier = PaddedGeometryAlignmentScorer


__all__ = [
    'DINO_ALIGNMENT_SCORE_SEMANTICS',
    'GroundingDinoAlignmentClassifier',
    'GroundingDinoAlignmentScorer',
    'GroundingDinoAlignmentVerifierBackend',
    'PaddedGeometryAlignmentClassifier',
    'PaddedGeometryAlignmentScorer',
]
