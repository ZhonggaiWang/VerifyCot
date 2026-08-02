"""Strict parsers for binary and four-way Qwen verifier JSON output."""

import json
from dataclasses import dataclass
from typing import Any, Dict

from .prompt import ROUTING_STATUSES


@dataclass(frozen=True)
class ParsedBinaryAlignmentOutput:
    aligned: bool
    confidence: float
    payload: Dict[str, Any]


@dataclass(frozen=True)
class ParsedRoutingOutput:
    status: str
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
    """Parse the minimal object-reference/image alignment decision."""

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


def parse_routing_output(text: str) -> ParsedRoutingOutput:
    """Parse one action-oriented four-way routing classification."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError('routing verifier output must be a non-empty string')
    payload = _first_json_object(text)
    status = payload.get('status')
    if not isinstance(status, str):
        raise ValueError('routing verifier status must be a string')
    status = status.strip().lower()
    if status not in ROUTING_STATUSES:
        raise ValueError(
            f'unknown routing status {status!r}; '
            f'expected one of {ROUTING_STATUSES}'
        )
    confidence = payload.get('confidence')
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
    ):
        raise ValueError('routing verifier confidence must be numeric')
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError('routing verifier confidence must be in [0, 1]')
    return ParsedRoutingOutput(
        status=status,
        confidence=confidence,
        payload=payload,
    )
