"""Canonical and legacy-compatible paths for experiment runs.

New experiments use one directory per logical run::

    output/<dataset>/runs/<split>/<study>/<method>/<setting>/<run_id>/

The older :func:`resolve_run_output` helper remains unchanged in behaviour so
existing command lines with an explicit ``--output`` continue to work.
"""

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple, Union


PathInput = Union[str, Path]
DEFAULT_OUTPUT_ROOT = Path('output')
DEFAULT_SETTING = 'default'
RUN_METADATA_SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _new_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _canonical_component(value: str, field_name: str) -> str:
    """Validate one portable directory component without renaming it."""
    if not isinstance(value, str):
        raise TypeError('{} must be a string'.format(field_name))
    if not value or value in {'.', '..'}:
        raise ValueError('{} must be a non-empty path component'.format(field_name))
    if '/' in value or '\\' in value:
        raise ValueError('{} must not contain path separators'.format(field_name))
    return value


@dataclass(frozen=True)
class RunLayout:
    """All standard paths belonging to one logical experiment run.

    Construct layouts with :func:`create_run_layout`.  Merely constructing a
    layout performs no filesystem writes.  :meth:`ensure_run_directories`
    creates the run directory and, optionally, ``artifacts/``; distributed
    ``shards/`` and stochastic ``repetitions/`` are deliberately always
    created by the runner that actually needs them.
    """

    dataset: str
    split: str
    study: str
    method: str
    setting: str
    run_id: str
    run_dir: Path
    results_path: Path
    layout_kind: str = 'canonical'

    @property
    def is_canonical(self) -> bool:
        return self.layout_kind == 'canonical'

    @property
    def is_exact_output(self) -> bool:
        return self.layout_kind == 'exact_output'

    @property
    def config_path(self) -> Path:
        return self.run_dir / 'run.config.json'

    @property
    def status_path(self) -> Path:
        return self.run_dir / 'run.status.json'

    @property
    def summary_path(self) -> Path:
        if self.is_canonical:
            return self.run_dir / 'results.summary.json'
        return self.results_path.with_suffix('.summary.json')

    @property
    def events_path(self) -> Path:
        return self.run_dir / 'verifier_events.jsonl'

    @property
    def log_path(self) -> Path:
        return self.run_dir / 'run.log'

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / 'artifacts'

    @property
    def shards_dir(self) -> Path:
        return self.run_dir / 'shards'

    @property
    def repetitions_dir(self) -> Path:
        return self.run_dir / 'repetitions'

    def ensure_run_directories(self, include_artifacts: bool = False) -> None:
        """Create only directories common to a normal, unsharded run."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if include_artifacts:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def identity(self) -> Dict[str, Any]:
        """Return the stable identity embedded in config and status files."""
        return {
            'schema_version': RUN_METADATA_SCHEMA_VERSION,
            'run_id': self.run_id,
            'dataset': self.dataset,
            'split': self.split,
            'study': self.study,
            'method': self.method,
            'setting': self.setting,
            'layout': self.layout_kind,
        }


def resolve_run_output(
        path: PathInput,
        run_id: Optional[str] = None) -> Tuple[Path, str]:
    """Put an explicit output file below a timestamp/run-id directory.

    This is the legacy path contract and intentionally does not impose the
    canonical dataset/study hierarchy.  For example, ``foo/results.jsonl``
    becomes ``foo/<run_id>/results.jsonl``.
    """
    resolved_run_id = run_id or _new_run_id()
    requested = Path(path)
    return requested.parent / resolved_run_id / requested.name, resolved_run_id


def create_run_layout(
        *,
        dataset: str,
        split: str,
        study: str,
        method: str,
        setting: str = DEFAULT_SETTING,
        run_id: Optional[str] = None,
        output: Optional[PathInput] = None,
        output_root: PathInput = DEFAULT_OUTPUT_ROOT) -> RunLayout:
    """Build a canonical layout or adapt an explicit legacy output path.

    Args:
        dataset: Dataset family, for example ``vstar`` or ``gqa``.
        split: Stable evaluated subset, for example ``full_238``.
        study: Experiment family such as ``routing`` or ``baseline``.
        method: Component combination being evaluated.
        setting: Short primary setting; full parameters belong in
            ``run.config.json``.
        run_id: Reusable run identifier.  Defaults to a local timestamp.
        output: Optional explicit results filename.  When supplied, it uses
            the legacy :func:`resolve_run_output` placement contract.
        output_root: Root of the canonical output tree; useful for tests and
            alternative workspaces.

    The function is side-effect free.  Call
    :meth:`RunLayout.ensure_run_directories` immediately before writing.
    """
    resolved_run_id = run_id or _new_run_id()
    if output is not None:
        results_path, resolved_run_id = resolve_run_output(
            output,
            resolved_run_id,
        )
        return RunLayout(
            dataset=dataset,
            split=split,
            study=study,
            method=method,
            setting=setting,
            run_id=resolved_run_id,
            run_dir=results_path.parent,
            results_path=results_path,
            layout_kind='timestamped_output',
        )

    components = (
        _canonical_component(dataset, 'dataset'),
        _canonical_component(split, 'split'),
        _canonical_component(study, 'study'),
        _canonical_component(method, 'method'),
        _canonical_component(setting, 'setting'),
        _canonical_component(resolved_run_id, 'run_id'),
    )
    run_dir = (
        Path(output_root)
        / components[0]
        / 'runs'
        / components[1]
        / components[2]
        / components[3]
        / components[4]
        / components[5]
    )
    return RunLayout(
        dataset=components[0],
        split=components[1],
        study=components[2],
        method=components[3],
        setting=components[4],
        run_id=components[5],
        run_dir=run_dir,
        results_path=run_dir / 'results.jsonl',
        layout_kind='canonical',
    )


def create_exact_output_layout(
        *,
        dataset: str,
        split: str,
        study: str,
        method: str,
        run_id: Optional[str],
        output: PathInput,
        setting: str = DEFAULT_SETTING) -> RunLayout:
    """Adapt an exact runner-owned output file without inserting a run-id.

    This boundary is intended for a launcher that has already resolved the
    canonical run and shard directory, notably GQA multi-GPU evaluation.  It
    differs deliberately from ``create_run_layout(output=...)``, which keeps
    the historical ``parent/<run_id>/filename`` behaviour.

    Distributed callers should pass their shared ``run_id`` so every shard
    records the same logical identity. A direct legacy call may omit it; in
    that case a timestamp is used only as metadata and is not inserted into
    the exact output path.
    """
    resolved_run_id = run_id or _new_run_id()
    identity = (
        _canonical_component(dataset, 'dataset'),
        _canonical_component(split, 'split'),
        _canonical_component(study, 'study'),
        _canonical_component(method, 'method'),
        _canonical_component(setting, 'setting'),
        _canonical_component(resolved_run_id, 'run_id'),
    )
    results_path = Path(output)
    if not results_path.name:
        raise ValueError('output must name a results file')
    return RunLayout(
        dataset=identity[0],
        split=identity[1],
        study=identity[2],
        method=identity[3],
        setting=identity[4],
        run_id=identity[5],
        run_dir=results_path.parent,
        results_path=results_path,
        layout_kind='exact_output',
    )


def _metadata_document(
        layout: RunLayout,
        payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError('run metadata payload must be a mapping')
    document = dict(payload)
    document.setdefault('resolved_paths', {
        'run_dir': str(layout.run_dir),
        'results': str(layout.results_path),
        'summary': str(layout.summary_path),
        'verifier_events': str(layout.events_path),
    })
    # The resolved layout is authoritative.  This prevents a copied config
    # template from silently claiming another run identity.
    document.update(layout.identity())
    return document


def _git_provenance() -> Dict[str, Any]:
    """Collect compact, best-effort source provenance for a run config."""
    result: Dict[str, Any] = {
        'python_executable': sys.executable,
        'host': platform.node(),
    }
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=normal'],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        result['git_commit'] = commit
        result['git_dirty'] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        result['git_commit'] = None
        result['git_dirty'] = None
    return result


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> Path:
    # Serialize before touching the filesystem, so invalid payloads cannot
    # leave an empty run directory or truncate an existing metadata file.
    serialized = json.dumps(
        document,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=str(path.parent),
        prefix='.' + path.name + '.',
        suffix='.tmp',
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary_path), str(path))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def write_run_config(
        layout: RunLayout,
        config: Mapping[str, Any]) -> Path:
    """Atomically write ``run.config.json`` with authoritative run identity."""
    document = dict(config)
    document.setdefault(
        'created_at',
        datetime.now().astimezone().isoformat(timespec='seconds'),
    )
    document.setdefault('provenance', _git_provenance())
    return _atomic_write_json(
        layout.config_path,
        _metadata_document(layout, document),
    )


def write_run_status(
        layout: RunLayout,
        status: str,
        **fields: Any) -> Path:
    """Atomically write the current lifecycle state to ``run.status.json``."""
    if not isinstance(status, str) or not status:
        raise ValueError('status must be a non-empty string')
    document = dict(fields)
    document['status'] = status
    document['updated_at'] = datetime.now().astimezone().isoformat(
        timespec='seconds'
    )
    return _atomic_write_json(
        layout.status_path,
        _metadata_document(layout, document),
    )


__all__ = [
    'DEFAULT_OUTPUT_ROOT',
    'DEFAULT_SETTING',
    'RUN_METADATA_SCHEMA_VERSION',
    'RunLayout',
    'create_exact_output_layout',
    'create_run_layout',
    'resolve_run_output',
    'write_run_config',
    'write_run_status',
]
