"""Timestamped output-path helpers for non-overwriting experiment runs."""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def resolve_run_output(path: str, run_id: Optional[str] = None) -> Tuple[Path, str]:
    """Put a requested output file below a shared timestamp/run-id directory."""
    resolved_run_id = run_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    requested = Path(path)
    return requested.parent / resolved_run_id / requested.name, resolved_run_id
