"""Grounding DINO relocation expert."""

from ...models.grounding_dino import (
    GroundingDinoBoxPredictor,
    GroundingDinoRunner,
)
from .predictor import PredictorGrounderBackend


class GroundingDinoGrounderBackend(PredictorGrounderBackend):
    def __init__(
            self,
            runner: GroundingDinoRunner,
            top_k_log: int = 20):
        super().__init__(
            predictor=GroundingDinoBoxPredictor(
                runner,
                top_k_log=top_k_log,
            ),
            source='grounding_dino_grounder',
        )
