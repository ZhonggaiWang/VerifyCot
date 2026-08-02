"""Versioned, model-neutral wire contract for Grounder workers.

Grounder endpoints operate on the immutable original image and return one
pixel-space box.  VoCoT-specific square-padding conversion belongs to the
consumer (for example :class:`RemoteGrounderBackend`), not to the worker.

The transport may add framing fields such as ``request_id`` and ``ok``.  The
parser intentionally ignores those fields while requiring every field in the
Grounder payload itself.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Tuple, cast


GROUNDER_OUTPUT_SCHEMA = 'vocot_grounder_output_v1'
ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM = (
    'absolute_xyxy_on_original_image'
)

PixelBox = Tuple[float, float, float, float]
ImageSize = Tuple[int, int]

_REQUIRED_FIELDS = frozenset({
    'grounder_output_schema',
    'available',
    'source',
    'coordinate_system',
    'bbox',
    'image_size',
    'confidence',
    'error',
    'metadata',
})


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty string')
    return value.strip()


def _image_size(value: Any) -> ImageSize:
    if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in value
            )):
        raise ValueError('image_size must be [width, height] integers')
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError('image_size must contain positive values')
    return width, height


def _pixel_box(value: Any, image_size: ImageSize) -> PixelBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('bbox must contain four coordinates')
    if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value):
        raise ValueError('bbox must contain only JSON numbers')
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box):
        raise ValueError('bbox coordinates must be finite')
    width, height = image_size
    if not (
            0.0 <= box[0] < box[2] <= float(width)
            and 0.0 <= box[1] < box[3] <= float(height)):
        raise ValueError(
            f'bbox {list(box)} is outside image {width}x{height}'
        )
    return cast(PixelBox, box)


def _confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('confidence must be a JSON number or null')
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError('confidence must be finite and in [0, 1]')
    return confidence


@dataclass(frozen=True)
class GrounderWireOutput:
    """Validated Grounder response independent of JSONL framing."""

    available: bool
    source: str
    coordinate_system: str
    bbox: Optional[PixelBox]
    image_size: ImageSize
    confidence: Optional[float]
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
            'available': self.available,
            'source': self.source,
            'coordinate_system': self.coordinate_system,
            'bbox': None if self.bbox is None else list(self.bbox),
            'image_size': list(self.image_size),
            'confidence': self.confidence,
            'error': self.error,
            'metadata': dict(self.metadata),
        }


def parse_grounder_output(payload: Mapping[str, Any]) -> GrounderWireOutput:
    """Strictly validate and normalize a canonical Grounder response."""

    if not isinstance(payload, Mapping):
        raise TypeError('grounder output must be a mapping')
    missing = sorted(_REQUIRED_FIELDS.difference(payload.keys()))
    if missing:
        raise ValueError(
            'grounder output is missing required field(s): '
            + ', '.join(missing)
        )
    if payload.get('grounder_output_schema') != GROUNDER_OUTPUT_SCHEMA:
        raise ValueError(
            'unsupported grounder_output_schema: '
            f'{payload.get("grounder_output_schema")!r}'
        )
    available = payload.get('available')
    if not isinstance(available, bool):
        raise ValueError('available must be boolean')
    source = _non_empty_string(payload.get('source'), 'source')
    coordinate_system = _non_empty_string(
        payload.get('coordinate_system'),
        'coordinate_system',
    )
    if coordinate_system != ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM:
        raise ValueError(
            'unsupported coordinate_system: '
            f'{coordinate_system!r}'
        )
    image_size = _image_size(payload.get('image_size'))
    confidence = _confidence(payload.get('confidence'))
    error = payload.get('error')
    if error is not None and (
            not isinstance(error, str) or not error.strip()):
        raise ValueError('error must be a non-empty string or null')
    if isinstance(error, str):
        error = error.strip()
    metadata = payload.get('metadata')
    if not isinstance(metadata, Mapping):
        raise ValueError('metadata must be a mapping')

    if available:
        bbox = _pixel_box(payload.get('bbox'), image_size)
        if error is not None:
            raise ValueError('available output must have error=null')
    else:
        if payload.get('bbox') is not None:
            raise ValueError('unavailable output must have bbox=null')
        if confidence is not None:
            raise ValueError('unavailable output must have confidence=null')
        if error is None:
            raise ValueError('unavailable output must provide an error')
        bbox = None

    return GrounderWireOutput(
        available=available,
        source=source,
        coordinate_system=coordinate_system,
        bbox=bbox,
        image_size=image_size,
        confidence=confidence,
        error=error,
        metadata=dict(metadata),
    )


def serialize_grounder_output(
        *,
        available: bool,
        source: str,
        bbox: Optional[PixelBox],
        image_size: ImageSize,
        confidence: Optional[float],
        error: Optional[str],
        metadata: Optional[Mapping[str, Any]] = None,
        coordinate_system: str = ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM,
) -> Dict[str, Any]:
    """Build a canonical response and validate it before transmission."""

    candidate = {
        'grounder_output_schema': GROUNDER_OUTPUT_SCHEMA,
        'available': available,
        'source': source,
        'coordinate_system': coordinate_system,
        'bbox': bbox,
        'image_size': image_size,
        'confidence': confidence,
        'error': error,
        'metadata': {} if metadata is None else dict(metadata),
    }
    return parse_grounder_output(candidate).as_dict()


__all__ = [
    'GROUNDER_OUTPUT_SCHEMA',
    'GrounderWireOutput',
    'ORIGINAL_IMAGE_PIXEL_COORDINATE_SYSTEM',
    'parse_grounder_output',
    'serialize_grounder_output',
]
