"""Benchmark labels and their archived four-way routing collapse."""

from typing import Dict, Tuple

from ...four_way.contracts import VerifierAction


# These are construction labels supplied by the controlled benchmark.  They
# remain useful for subtype reporting even though the active learned verifier
# interfaces are binary alignment and four-way routing.
CONTROLLED_STATUSES: Tuple[str, ...] = (
    'aligned',
    'wrong_object',
    'partial_coverage',
    'ambiguous',
    'unsupported',
)

# ``unsupported`` was collapsed to relocate in the completed four-way
# experiments.  Production policy may instead abstain; this mapping is kept
# benchmark-local so it cannot be mistaken for runtime routing policy.
CONTROLLED_STATUS_TO_ROUTING_ACTION: Dict[str, VerifierAction] = {
    'aligned': 'no_action',
    'wrong_object': 'relocate',
    'unsupported': 'relocate',
    'partial_coverage': 'expand',
    'ambiguous': 'tighten',
}


__all__ = [
    'CONTROLLED_STATUSES',
    'CONTROLLED_STATUS_TO_ROUTING_ACTION',
]
