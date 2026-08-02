"""Model-agnostic, line-delimited JSON worker transport."""

from contextlib import redirect_stdout
import json
import sys
import traceback
from typing import Any, Iterable, Mapping, Optional, Protocol, TextIO


DEFAULT_PROTOCOL_NAME = 'vocot_worker_v1'
DEFAULT_RESPONSE_PREFIX = '@@VOCOT_WORKER_JSON@@'


class WorkerRequestError(ValueError):
    """A malformed or unsatisfied request that must not kill the worker."""


class WorkerEngine(Protocol):
    def handle(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


def optional_request_id(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get('request_id')
    return value if isinstance(value, str) and value else None


def process_request_line(
        line: str,
        engine: WorkerEngine,
        stderr: TextIO,
        protocol_name: str = DEFAULT_PROTOCOL_NAME,
) -> dict:
    """Parse and execute one line, isolating errors and model stdout."""

    payload: Any = None
    try:
        payload = json.loads(line)
        request_id = optional_request_id(payload)
        with redirect_stdout(stderr):
            result = dict(engine.handle(payload))
        return {
            'protocol': protocol_name,
            'request_id': request_id,
            'ok': True,
            **result,
        }
    except Exception as error:
        traceback.print_exc(file=stderr)
        return {
            'protocol': protocol_name,
            'request_id': optional_request_id(payload),
            'ok': False,
            'error_type': type(error).__name__,
            'error': str(error),
        }


def serve_jsonl(
        engine: WorkerEngine,
        stdin: Iterable[str] = sys.stdin,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
        protocol_name: str = DEFAULT_PROTOCOL_NAME,
        response_prefix: str = DEFAULT_RESPONSE_PREFIX,
) -> int:
    """Serve requests sequentially until EOF or acknowledged shutdown."""

    for line in stdin:
        if not line.strip():
            continue
        response = process_request_line(
            line,
            engine,
            stderr,
            protocol_name=protocol_name,
        )
        stdout.write(
            response_prefix
            + json.dumps(response, ensure_ascii=False, allow_nan=False)
            + '\n'
        )
        stdout.flush()
        if response.get('ok') and response.get('shutdown'):
            return 0
    return 0


__all__ = [
    'DEFAULT_PROTOCOL_NAME',
    'DEFAULT_RESPONSE_PREFIX',
    'WorkerEngine',
    'WorkerRequestError',
    'optional_request_id',
    'process_request_line',
    'serve_jsonl',
]
