"""Oracle lookup backends used only by archived prompt-repair experiments."""

from .single_candidate import SingleCandidateOracleVerifier
from .stored import StoredOracleVerifier

__all__ = [
    'SingleCandidateOracleVerifier',
    'StoredOracleVerifier',
]
