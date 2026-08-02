"""Strict parser owned by the standalone Qwen binary verifier path."""

import json
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ParsedBinaryAlignmentOutput:
    aligned: bool
    confidence: float
    payload: Dict[str, Any]


def _first_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != '{':
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError('verifier output does not contain a JSON object')


def parse_binary_alignment_output(text: str) -> ParsedBinaryAlignmentOutput:
    """Parse one binary label and confidence-in-emitted-label value."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError('verifier output must be a non-empty string')
    payload = _first_json_object(text)
    aligned = payload.get('aligned')
    if not isinstance(aligned, bool):
        raise ValueError('binary verifier aligned field must be true or false')
    confidence = payload.get('confidence')
    if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
    ):
        raise ValueError('binary verifier confidence must be numeric')
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError('binary verifier confidence must be in [0, 1]')
    return ParsedBinaryAlignmentOutput(
        aligned=aligned,
        confidence=confidence,
        payload=payload,
    )


__all__ = [
    'ParsedBinaryAlignmentOutput',
    'parse_binary_alignment_output',
]
