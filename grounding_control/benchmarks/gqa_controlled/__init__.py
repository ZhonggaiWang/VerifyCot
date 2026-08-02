"""Adapter and metrics for the controlled GQA verifier benchmark."""

from .adapter import (
    GQAControlledExample,
    expected_status_from_record,
    load_examples,
)
from .labels import (
    CONTROLLED_STATUSES,
    CONTROLLED_STATUS_TO_ROUTING_ACTION,
)
from .metrics import (
    compute_binary_alignment_metrics,
    compute_routing_metrics,
)

__all__ = [
    'compute_binary_alignment_metrics',
    'compute_routing_metrics',
    'CONTROLLED_STATUSES',
    'CONTROLLED_STATUS_TO_ROUTING_ACTION',
    'expected_status_from_record',
    'GQAControlledExample',
    'load_examples',
]
