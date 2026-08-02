"""Qwen2.5-VL relocation expert."""

from ...models.qwen25_vl import Qwen25VLBoxPredictor, Qwen25VLRunner
from .predictor import PredictorGrounderBackend


class Qwen25VLGrounderBackend(PredictorGrounderBackend):
    def __init__(self, runner: Qwen25VLRunner, **predictor_kwargs):
        super().__init__(
            predictor=Qwen25VLBoxPredictor(
                runner,
                **predictor_kwargs,
            ),
            source='qwen25_vl_grounder',
        )
