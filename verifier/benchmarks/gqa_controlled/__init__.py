"""Adapter and metrics for the controlled five-way GQA verifier benchmark."""

from .adapter import (
    GQAControlledExample,
    expected_status_from_record,
    load_examples,
)
from .metrics import (
    compute_binary_alignment_metrics,
    compute_routing_metrics,
    compute_verifier_metrics,
)

__all__ = [
    'compute_verifier_metrics',
    'compute_binary_alignment_metrics',
    'compute_routing_metrics',
    'expected_status_from_record',
    'GQAControlledExample',
    'load_examples',
]
