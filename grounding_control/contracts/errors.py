"""Role-level expert errors shared without depending on orchestration code."""

from typing import Any, Dict, Optional


class ExpertNotConfiguredError(RuntimeError):
    """A routing policy requested an expert that was not configured."""


class ExpertUnavailableError(RuntimeError):
    """A configured expert cannot act on this particular request."""

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.metadata = dict(metadata or {})


class VerifierFailClosedError(RuntimeError):
    """A verifier failure that must abort the current inference.

    Ordinary verifier exceptions are intentionally converted to an unknown
    alignment result by the pre-commit controller.  Backends configured for
    fail-closed behavior raise this explicit boundary exception instead so
    the controller cannot silently turn that configuration back into
    fail-open routing.
    """

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.metadata = dict(metadata or {})


__all__ = [
    'ExpertNotConfiguredError',
    'ExpertUnavailableError',
    'VerifierFailClosedError',
]
