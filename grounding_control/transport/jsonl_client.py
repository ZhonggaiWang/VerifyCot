"""Synchronous client for a persistent line-delimited JSON worker."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence
import uuid

from .jsonl_protocol import (
    DEFAULT_PROTOCOL_NAME,
    DEFAULT_RESPONSE_PREFIX,
)


class WorkerClientError(RuntimeError):
    """Base error raised by the persistent worker client."""


class WorkerTimeoutError(WorkerClientError):
    """The worker did not return a response before the request deadline."""


class RemoteWorkerError(WorkerClientError):
    """The worker returned a structured ``ok=false`` response."""

    def __init__(
            self,
            message: str,
            error_type: Optional[str] = None,
            response: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.response = dict(response or {})


class PersistentJsonlWorkerClient:
    """Own one worker subprocess and serialize request/response exchanges."""

    def __init__(
            self,
            command: Sequence[str],
            *,
            cwd: Optional[str] = None,
            env: Optional[Mapping[str, str]] = None,
            timeout: float = 300.0,
            protocol_name: str = DEFAULT_PROTOCOL_NAME,
            response_prefix: str = DEFAULT_RESPONSE_PREFIX,
            stderr=None,
            start: bool = True):
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError('command must be a non-empty argument sequence')
        if float(timeout) <= 0:
            raise ValueError('timeout must be positive')
        self.command = [str(value) for value in command]
        self.cwd = None if cwd is None else str(Path(cwd))
        self.env_overrides = {
            str(key): str(value) for key, value in dict(env or {}).items()
        }
        self.timeout = float(timeout)
        self.protocol_name = str(protocol_name)
        self.response_prefix = str(response_prefix)
        self.stderr = stderr
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._closed = False
        if start:
            self.start()

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process

    def start(self) -> None:
        if self._closed:
            raise WorkerClientError('worker client is already closed')
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise WorkerClientError(
                f'worker already exited with code {self._process.returncode}'
            )
        environment = os.environ.copy()
        environment.update(self.env_overrides)
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise WorkerClientError('failed to open worker stdin/stdout pipes')

    def _read_response(
            self,
            request_id: str,
            timeout: float) -> Dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise WorkerClientError('worker is not running')
        deadline = time.monotonic() + timeout
        unexpected_lines = []
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if process.poll() is not None:
                    raise WorkerClientError(
                        f'worker exited with code {process.returncode}; '
                        f'unexpected stdout={unexpected_lines[-5:]!r}'
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerTimeoutError(
                        f'worker request {request_id!r} timed out after '
                        f'{timeout:.1f}s'
                    )
                if not selector.select(remaining):
                    raise WorkerTimeoutError(
                        f'worker request {request_id!r} timed out after '
                        f'{timeout:.1f}s'
                    )
                line = process.stdout.readline()
                if line == '':
                    raise WorkerClientError(
                        'worker stdout closed before a response was received'
                    )
                line = line.rstrip('\r\n')
                if not line.startswith(self.response_prefix):
                    unexpected_lines.append(line)
                    continue
                try:
                    response = json.loads(line[len(self.response_prefix):])
                except json.JSONDecodeError as error:
                    raise WorkerClientError(
                        f'worker returned invalid JSON: {line!r}'
                    ) from error
                if response.get('protocol') != self.protocol_name:
                    raise WorkerClientError(
                        'worker response has an unexpected protocol: '
                        f'{response.get("protocol")!r}'
                    )
                if response.get('request_id') != request_id:
                    raise WorkerClientError(
                        'worker response request_id mismatch: '
                        f'{response.get("request_id")!r} != {request_id!r}'
                    )
                return dict(response)
        finally:
            selector.close()

    def request(
            self,
            payload: Mapping[str, Any],
            *,
            timeout: Optional[float] = None) -> Dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise TypeError('worker payload must be a mapping')
        with self._lock:
            if self._closed:
                raise WorkerClientError('worker client is closed')
            self.start()
            process = self._process
            if process is None or process.stdin is None:
                raise WorkerClientError('worker stdin is unavailable')
            request_id = str(payload.get('request_id') or uuid.uuid4().hex)
            request = dict(payload)
            request['protocol'] = self.protocol_name
            request['request_id'] = request_id
            try:
                process.stdin.write(
                    json.dumps(
                        request,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + '\n'
                )
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise WorkerClientError(
                    'failed to write request to worker'
                ) from error
            response = self._read_response(
                request_id,
                self.timeout if timeout is None else float(timeout),
            )
            if not response.get('ok'):
                raise RemoteWorkerError(
                    str(response.get('error') or 'remote worker failed'),
                    error_type=response.get('error_type'),
                    response=response,
                )
            return response

    def ping(self, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self.request({'operation': 'ping'}, timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            process = self._process
            if process is not None and process.poll() is None:
                try:
                    # Inline the exchange because request() would reacquire
                    # the non-reentrant lock.
                    request_id = uuid.uuid4().hex
                    assert process.stdin is not None
                    process.stdin.write(json.dumps({
                        'protocol': self.protocol_name,
                        'request_id': request_id,
                        'operation': 'shutdown',
                    }) + '\n')
                    process.stdin.flush()
                    self._read_response(
                        request_id,
                        min(self.timeout, 10.0),
                    )
                except Exception:
                    process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()
            self._closed = True

    def __enter__(self) -> 'PersistentJsonlWorkerClient':
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    'PersistentJsonlWorkerClient',
    'RemoteWorkerError',
    'WorkerClientError',
    'WorkerTimeoutError',
]
