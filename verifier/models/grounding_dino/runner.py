"""Reusable lazy Hugging Face Grounding DINO model runner.

Heavy imports are deliberately delayed until the first real inference call so
the original VoCoT environment can still import evaluator and test modules.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, List, Protocol, Tuple

from PIL import Image


PixelBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class GroundingDinoDetection:
    """One detector output in the original image's absolute pixel frame."""

    box_original_pixel_xyxy: PixelBox
    score: float
    label: str


class GroundingDinoRunner(Protocol):
    """Minimal detector contract used by the geometry classifier."""

    def detect(
            self,
            image: Image.Image,
            object_reference: str,
    ) -> List[GroundingDinoDetection]:
        """Localize ``object_reference`` without seeing the candidate box."""


def normalize_grounding_query(object_reference: str) -> str:
    """Apply only the normalization required by text-conditioned detection."""

    if not isinstance(object_reference, str):
        raise TypeError('object_reference must be a string')
    query = ' '.join(object_reference.strip().lower().split())
    if not query:
        raise ValueError('object_reference must not be empty')
    if not query.endswith('.'):
        query += '.'
    return query


class LocalGroundingDinoRunner:
    """Run a local Transformers Grounding DINO checkpoint."""

    def __init__(
            self,
            model_path: str,
            device: str = 'cuda:0',
            dtype: str = 'float32',
            box_threshold: float = 0.3,
            text_threshold: float = 0.25,
            local_files_only: bool = True):
        if not model_path:
            raise ValueError('model_path is required')
        for value, name in (
            (box_threshold, 'box_threshold'),
            (text_threshold, 'text_threshold'),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f'{name} must be finite and in [0, 1]')
        self.model_path = str(model_path)
        self.device = str(device)
        self.dtype = str(dtype)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.local_files_only = bool(local_files_only)
        self.last_run_metadata: Dict[str, Any] = {}
        self._model = None
        self._processor = None
        self._input_device = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except ImportError as error:
            raise RuntimeError(
                'LocalGroundingDinoRunner requires a Transformers environment '
                'with Grounding DINO support (the project qwen25 environment '
                'uses Transformers 4.49). Do not upgrade the original VoCoT '
                'generator environment.'
            ) from error

        dtype_by_name = {
            'bfloat16': torch.bfloat16,
            'bf16': torch.bfloat16,
            'float16': torch.float16,
            'fp16': torch.float16,
            'float32': torch.float32,
            'fp32': torch.float32,
        }
        if self.dtype == 'auto':
            torch_dtype = 'auto'
        else:
            try:
                torch_dtype = dtype_by_name[self.dtype.lower()]
            except KeyError as error:
                raise ValueError(
                    f'unsupported Grounding DINO dtype: {self.dtype!r}'
                ) from error

        load_kwargs: Dict[str, Any] = {
            'torch_dtype': torch_dtype,
            'local_files_only': self.local_files_only,
        }
        if self.device == 'auto' or self.device.startswith('cuda'):
            load_kwargs['device_map'] = self.device
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if self.device != 'auto' and not self.device.startswith('cuda'):
            model.to(self.device)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=self.local_files_only,
        )

        self._torch = torch
        self._model = model
        self._processor = processor
        self._input_device = next(model.parameters()).device

    def _synchronize(self) -> None:
        if (
            self._torch is not None
            and self._input_device is not None
            and self._input_device.type == 'cuda'
        ):
            self._torch.cuda.synchronize(self._input_device)

    def detect(
            self,
            image: Image.Image,
            object_reference: str,
    ) -> List[GroundingDinoDetection]:
        if not isinstance(image, Image.Image):
            raise TypeError('image must be a PIL.Image.Image')
        if image.width <= 0 or image.height <= 0:
            raise ValueError('image must be non-empty')
        query = normalize_grounding_query(object_reference)
        self._load()

        started = time.perf_counter()
        preprocess_started = started
        inputs = self._processor(
            images=image.convert('RGB'),
            text=query,
            return_tensors='pt',
        )
        inputs = inputs.to(self._input_device)
        preprocess_finished = time.perf_counter()

        self._synchronize()
        forward_started = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        self._synchronize()
        forward_finished = time.perf_counter()

        postprocess_started = time.perf_counter()
        processed = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = processed.get('boxes', [])
        scores = processed.get('scores', [])
        labels = processed.get('text_labels')
        if labels is None:
            labels = processed.get('labels', [])
        detections: List[GroundingDinoDetection] = []
        for box, score, label in zip(boxes, scores, labels):
            box_values = box.detach().cpu().tolist()
            score_value = float(score.detach().cpu().item())
            detections.append(GroundingDinoDetection(
                box_original_pixel_xyxy=tuple(
                    float(value) for value in box_values
                ),
                score=score_value,
                label=str(label),
            ))
        postprocess_finished = time.perf_counter()
        self.last_run_metadata = {
            'grounding_query': query,
            'timing_ms': {
                'preprocess': (
                    (preprocess_finished - preprocess_started) * 1000.0
                ),
                'forward': (
                    (forward_finished - forward_started) * 1000.0
                ),
                'postprocess': (
                    (postprocess_finished - postprocess_started) * 1000.0
                ),
                'total': (postprocess_finished - started) * 1000.0,
            },
        }
        return detections
