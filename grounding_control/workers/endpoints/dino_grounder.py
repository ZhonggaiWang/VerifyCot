"""JSON request adapter for a Grounding DINO relocation expert."""

from typing import Any, Dict, Mapping

from ...models import BoxPredictionRequest
from ...models.grounding_dino import GroundingDinoBoxPredictor
from ...transport.grounder_wire import serialize_grounder_output
from ...transport.request_io import load_image, required_string


DINO_GROUNDER_SOURCE = 'grounding_dino_grounder'


class DinoGrounderEndpoint:
    def __init__(self, predictor: GroundingDinoBoxPredictor):
        self.predictor = predictor

    def handle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        image = load_image(payload.get('image_path'))
        reference = required_string(payload, 'object_reference')
        prediction = self.predictor.predict(BoxPredictionRequest(
            image=image,
            object_reference=reference,
            sample_id=str(payload.get('sample_id') or ''),
        ))
        metadata = dict(prediction.metadata or {})
        detections = []
        for record in metadata.get('valid_detections', []):
            detections.append({
                'original_index': record['original_index'],
                'bbox_original_pixel_xyxy': (
                    record['box_original_pixel_xyxy']
                ),
                'raw_bbox_original_pixel_xyxy': (
                    record['raw_box_original_pixel_xyxy']
                ),
                'boundary_clipped': record['boundary_clipped'],
                'score': record['score'],
                'label': record['label'],
            })
        # Preserve every field exposed by the pre-v1 DINO endpoint under one
        # namespaced metadata record.  New consumers use only the canonical
        # wire fields, while archived diagnostics remain fully inspectable.
        metadata['legacy_dino_endpoint_v0'] = {
            'bbox_original_pixel_xyxy': (
                None if prediction.bbox_pixel_xyxy is None
                else list(prediction.bbox_pixel_xyxy)
            ),
            'score': prediction.confidence,
            'label': metadata.get('selected_grounding_label'),
            'selected_detection_index': metadata.get(
                'selected_detection_index'
            ),
            'selection_policy': 'highest_detector_score',
            'image_size': list(image.size),
            'detection_count': metadata.get('detection_count'),
            'valid_detection_count': metadata.get(
                'valid_detection_count'
            ),
            'detections': detections,
            'metadata': dict(metadata.get('runner_metadata') or {}),
        }
        available = prediction.bbox_pixel_xyxy is not None
        return serialize_grounder_output(
            available=available,
            source=DINO_GROUNDER_SOURCE,
            bbox=prediction.bbox_pixel_xyxy,
            image_size=image.size,
            confidence=prediction.confidence if available else None,
            error=(
                None if available else
                prediction.error or 'no_valid_grounding_detection'
            ),
            metadata=metadata,
        )


__all__ = ['DINO_GROUNDER_SOURCE', 'DinoGrounderEndpoint']
