"""Compatibility import for the archived stored-oracle repair backend.

New code should import this class from ``verifier.legacy``.
"""

from .legacy.oracle_backends import StoredOracleVerifier

__all__ = ['StoredOracleVerifier']
