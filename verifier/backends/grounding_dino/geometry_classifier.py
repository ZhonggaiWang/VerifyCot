"""Route a candidate box using an independent Grounding DINO localization."""

from dataclasses import asdict
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..qwen25_vl.action_classifier import GroundingActionInput
from ..qwen25_vl.grounding_classifier import GroundingGeometryLookup
from ..qwen25_vl.grounding_geometry import route_from_grounding_geometry
from .runner import (
    GroundingDinoDetection,
    GroundingDinoRunner,
    normalize_grounding_query,
)


PixelBox = Tuple[float, float, float, float]


def _json_safe_box(values: Sequence[float]) -> List[Any]:
    result: List[Any] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            result.append(str(value))
        else:
            result.append(numeric if math.isfinite(numeric) else None)
    return result


def _validated_candidate_box(
        values: Sequence[float],
        image_size: Tuple[int, int],
) -> PixelBox:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError('candidate pixel bbox must contain four coordinates')
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f'candidate pixel bbox is non-finite: {box}')
    width, height = image_size
    if not (
        0.0 <= box[0] < box[2] <= width
        and 0.0 <= box[1] < box[3] <= height
    ):
        raise ValueError(
            f'candidate pixel bbox is outside image {image_size}: {box}'
        )
    return box  # type: ignore[return-value]


def _clip_detection_box(
        values: Sequence[float],
        image_size: Tuple[int, int],
) -> Tuple[Optional[PixelBox], List[str], Optional[str]]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None, [], 'box_must_contain_four_coordinates'
    try:
        raw = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None, [], 'box_coordinates_must_be_numeric'
    if not all(math.isfinite(value) for value in raw):
        return None, [], 'box_coordinates_are_non_finite'

    width, height = image_size
    clipped = (
        min(max(raw[0], 0.0), float(width)),
        min(max(raw[1], 0.0), float(height)),
        min(max(raw[2], 0.0), float(width)),
        min(max(raw[3], 0.0), float(height)),
    )
    side_names = ('x1', 'y1', 'x2', 'y2')
    clipped_sides = [
        side for side, before, after in zip(side_names, raw, clipped)
        if before != after
    ]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None, clipped_sides, 'box_is_empty_after_clipping'
    return clipped, clipped_sides, None


class GroundingDinoGeometryClassifier:
    """Use Grounding DINO as a candidate-hidden reference localizer."""

    def __init__(
            self,
            runner: GroundingDinoRunner,
            accept_iou_threshold: float = 0.5,
            containment_threshold: float = 0.7,
            top_k_log: int = 20):
        if not 0.0 < accept_iou_threshold <= 1.0:
            raise ValueError('accept_iou_threshold must be in (0, 1]')
        if not 0.0 < containment_threshold <= 1.0:
            raise ValueError('containment_threshold must be in (0, 1]')
        if (
            not isinstance(top_k_log, int)
            or isinstance(top_k_log, bool)
            or top_k_log <= 0
        ):
            raise ValueError('top_k_log must be a positive integer')
        self.runner = runner
        self.accept_iou_threshold = float(accept_iou_threshold)
        self.containment_threshold = float(containment_threshold)
        self.top_k_log = int(top_k_log)

    def classify(
            self,
            candidate: GroundingActionInput,
            image_mode: str = 'raw_image',
    ) -> GroundingGeometryLookup:
        if image_mode != 'raw_image':
            raise ValueError(
                'Grounding DINO geometry supports only raw_image because the '
                'candidate must remain hidden from the independent localizer'
            )
        image_size = (candidate.image.width, candidate.image.height)
        candidate_box = _validated_candidate_box(
            candidate.candidate_bbox_pixel_xyxy,
            image_size,
        )
        query = normalize_grounding_query(candidate.object_reference)
        started = time.perf_counter()
        detections = self.runner.detect(
            candidate.image,
            candidate.object_reference,
        )
        total_ms = (time.perf_counter() - started) * 1000.0

        raw_detection_records: List[Dict[str, Any]] = []
        valid_detections = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, GroundingDinoDetection):
                raise TypeError(
                    'GroundingDinoRunner.detect() must return '
                    'GroundingDinoDetection values'
                )
            clipped_box, clipped_sides, rejection = _clip_detection_box(
                detection.box_original_pixel_xyxy,
                image_size,
            )
            score = float(detection.score)
            if not math.isfinite(score):
                rejection = 'detection_score_is_non_finite'
            elif not 0.0 <= score <= 1.0:
                rejection = 'detection_score_is_outside_0_1'
            record = {
                'original_index': index,
                'raw_box_original_pixel_xyxy': _json_safe_box(
                    detection.box_original_pixel_xyxy
                ),
                'box_original_pixel_xyxy': (
                    list(clipped_box) if clipped_box is not None else None
                ),
                'score': score,
                'label': detection.label,
                'boundary_clipped': bool(clipped_sides),
                'boundary_clipped_sides': clipped_sides,
                'valid': rejection is None,
                'rejection_reason': rejection,
            }
            raw_detection_records.append(record)
            if rejection is None:
                valid_detections.append((index, detection, clipped_box))

        # Python's sort is stable, so equal scores preserve detector order.
        valid_detections.sort(key=lambda item: -float(item[1].score))
        runner_metadata = dict(
            getattr(self.runner, 'last_run_metadata', {}) or {}
        )
        timing = runner_metadata.get('timing_ms')
        if not isinstance(timing, dict):
            timing = {'total': total_ms}
        metadata: Dict[str, Any] = {
            'backend': 'grounding_dino_geometry_router_raw_image',
            'image_mode': image_mode,
            'model_image_count': 1,
            'sample_id': candidate.sample_id,
            'model_path': getattr(self.runner, 'model_path', None),
            'coordinate_system': 'absolute_xyxy_on_original_image',
            'original_image_size': list(image_size),
            'object_reference': candidate.object_reference,
            'grounding_query': runner_metadata.get(
                'grounding_query',
                query,
            ),
            'candidate_original_pixel_bbox_xyxy': list(candidate_box),
            'box_threshold': getattr(
                self.runner,
                'box_threshold',
                None,
            ),
            'text_threshold': getattr(
                self.runner,
                'text_threshold',
                None,
            ),
            'selection_policy': 'highest_detector_score_candidate_hidden',
            'confidence_semantics': (
                'detector_score_not_calibrated_routing_probability'
            ),
            'detection_count': len(detections),
            'valid_detection_count': len(valid_detections),
            'invalid_detection_count': (
                len(detections) - len(valid_detections)
            ),
            'detections_logged_count': min(
                len(raw_detection_records),
                self.top_k_log,
            ),
            'detections': raw_detection_records[:self.top_k_log],
            'selected_detection_index': None,
            'selected_grounding_box_original_pixel_xyxy': None,
            'selected_grounding_score': None,
            'selected_grounding_label': None,
            'localization_failed': False,
            'parse_failed': False,
            'geometry': None,
            'timing_ms': timing,
        }
        if not valid_detections:
            metadata['localization_failed'] = True
            metadata['parse_failed'] = True
            return GroundingGeometryLookup(
                status=None,
                confidence=None,
                error='no_valid_grounding_detection',
                metadata=metadata,
            )

        selected_index, selected, selected_box = valid_detections[0]
        decision = route_from_grounding_geometry(
            candidate_box,
            selected_box,
            accept_iou_threshold=self.accept_iou_threshold,
            containment_threshold=self.containment_threshold,
        )
        metadata['selected_detection_index'] = selected_index
        metadata['selected_grounding_box_original_pixel_xyxy'] = list(
            selected_box
        )
        metadata['selected_grounding_score'] = float(selected.score)
        metadata['selected_grounding_label'] = selected.label
        metadata['geometry'] = asdict(decision)
        return GroundingGeometryLookup(
            status=decision.action,
            confidence=float(selected.score),
            error=None,
            metadata=metadata,
        )
