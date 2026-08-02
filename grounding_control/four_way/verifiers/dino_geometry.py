"""Grounding DINO plus deterministic geometry for four-action routing."""

from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from ..contracts import ActionVerifierBackend, ActionVerifierOutput
from ...contracts.verifier import VerificationRequest
from ...models.grounding_dino import (
    GroundingDinoBoxPredictor,
    GroundingDinoRunner,
)
from ...verifiers.box_geometry import (
    GeometryVerificationInput,
    GeometryVerificationLookup,
    PaddedGeometryVerificationInput,
)
from .geometry import GeometryVerifier, PaddedGeometryVerifier


class GroundingDinoGeometryClassifier:
    """Use DINO box prediction and archived four-action geometry policy."""

    def __init__(
            self,
            runner: GroundingDinoRunner,
            accept_iou_threshold: float = 0.5,
            containment_threshold: float = 0.7,
            top_k_log: int = 20):
        self.runner = runner
        self.accept_iou_threshold = float(accept_iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.top_k_log = int(top_k_log)
        self.predictor = GroundingDinoBoxPredictor(
            runner,
            top_k_log=top_k_log,
        )
        self.verifier = GeometryVerifier(
            self.predictor,
            accept_iou_threshold=accept_iou_threshold,
            containment_threshold=containment_threshold,
            backend_name='grounding_dino_geometry_router_raw_image',
        )
        self.padded_verifier = PaddedGeometryVerifier(
            self.predictor,
            accept_iou_threshold=accept_iou_threshold,
            containment_threshold=containment_threshold,
            backend_name=(
                'grounding_dino_geometry_router_'
                'vocot_padded_normalized'
            ),
        )

    def classify(
            self,
            candidate: GeometryVerificationInput,
            image_mode: str = 'raw_image',
    ) -> GeometryVerificationLookup:
        return self.verifier.classify(candidate, image_mode=image_mode)

    def classify_action(
            self,
            candidate: GeometryVerificationInput,
            image_mode: str = 'raw_image',
    ) -> ActionVerifierOutput:
        return self.verifier.classify_action(
            candidate,
            image_mode=image_mode,
        )

    def classify_padded_action(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> ActionVerifierOutput:
        return self.padded_verifier.classify_action(candidate)


class PaddedGeometryActionClassifier(Protocol):
    def classify_padded_action(
            self,
            candidate: PaddedGeometryVerificationInput,
    ) -> ActionVerifierOutput:
        ...


def _request_image(request: VerificationRequest) -> Image.Image:
    image: Any = request.sample_context.get('image')
    if isinstance(image, Image.Image):
        return image.convert('RGB').copy()
    image_path = request.sample_context.get('image_path')
    if isinstance(image_path, (str, Path)):
        path = Path(image_path)
        if path.is_file():
            with Image.open(path) as opened:
                return opened.convert('RGB').copy()
    raise ValueError(
        'DINO verifier requires sample_context["image"] as a PIL image or '
        'sample_context["image_path"] as a readable path'
    )


class GroundingDinoGeometryVerifierBackend(ActionVerifierBackend):
    """Judge online VoCoT candidates using DINO plus box geometry."""

    def __init__(self, classifier: PaddedGeometryActionClassifier):
        self.classifier = classifier

    def verify_action(
            self,
            request: VerificationRequest,
    ) -> ActionVerifierOutput:
        return self.classifier.classify_padded_action(
            PaddedGeometryVerificationInput(
                image=_request_image(request),
                object_reference=request.object_reference,
                candidate_bbox_padded_normalized_xyxy=(
                    request.candidate_bbox
                ),
                sample_id=request.sample_id,
            )
        )


__all__ = [
    'GroundingDinoGeometryClassifier',
    'GroundingDinoGeometryVerifierBackend',
    'PaddedGeometryActionClassifier',
]
