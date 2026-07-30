"""Compatibility import for the archived single-candidate oracle backend.

New code should import this class from ``verifier.legacy``.
"""

from .legacy.oracle_backends import SingleCandidateOracleVerifier

__all__ = ['SingleCandidateOracleVerifier']
