"""Candidate-aware, model-visible input shared by Qwen verifier protocols."""

from dataclasses import dataclass

from PIL import Image

from ...contracts.boxes import Box
from .rendering import COORDINATE_SYSTEM


@dataclass(frozen=True)
class CandidateVerificationInput:
    """Input visible to Qwen without labels, GT, or generation traces."""

    image: Image.Image
    object_reference: str
    candidate_bbox: Box
    sample_id: str = ''
    coordinate_system: str = COORDINATE_SYSTEM


__all__ = ['CandidateVerificationInput']
