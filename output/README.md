# Experiment output layout

Historical output is frozen in place.  Every new experiment should use one
canonical directory per logical run:

```text
output/<dataset>/runs/<split>/<study>/<method>/<setting>/<run_id>/
```

For example:

```text
output/vstar/runs/full_238/routing/
  dino_geometry__oracle_experts/iou_0p5/20260801_153000__main/
```

The path dimensions have stable meanings:

- `dataset`: `vstar`, `gqa`, or `verifier_benchmark`.
- `split`: the exact evaluated population, such as `full_238`,
  `full_238_matchable_198`, `val_1000_dev`, `testdev_12578`, or
  `gqa_controlled_v1_dev`.
- `study`: the comparison family, such as `baseline`, `counterfactual`,
  `oracle`, `routing`, `repair`, or `compatibility`.
- `method`: the component combination, for example
  `dino_geometry__oracle_experts` or `matched_random__oracle_experts`.
- `setting`: one short primary variant such as `iou_0p5`; use `default` when
  there is no meaningful variant.
- `run_id`: normally `<timestamp>` or `<timestamp>__<tag>`. Complete parameters
  belong in the config, not in an increasingly long directory name.

Use lower-case `snake_case`; encode decimal points as `p` in new path names
(`iou_0p5`).

## Files in one run

```text
<run_id>/
  run.config.json
  run.status.json
  results.jsonl
  results.summary.json
  verifier_events.jsonl
  run.log
  artifacts/
  shards/          # only when one logical run is data-parallel
  repetitions/     # only for independent random repetitions
```

The root `results.jsonl` and `results.summary.json` are the authoritative
outputs used for comparisons.  Optional debug renders, plots, and traces go
under `artifacts/`; do not add new top-level filenames for every backend.

`run.config.json` should include model/backend configuration, input manifests,
coordinate system, thresholds, seed, command, Git commit, and dirty-worktree
state.  `run.status.json` is the small mutable lifecycle record (`running`,
`completed`, `completed_with_errors`, or `failed`).  Both files contain schema
version and run identity.

The path and metadata helpers live in `grounding_control/run_paths.py`:

```python
from grounding_control.run_paths import (
    create_run_layout,
    write_run_config,
    write_run_status,
)

layout = create_run_layout(
    dataset='vstar',
    split='full_238',
    study='routing',
    method='dino_geometry__oracle_experts',
    setting='iou_0p5',
    run_id=args.run_id,
)
layout.ensure_run_directories()
write_run_config(layout, config)
write_run_status(layout, 'running')
```

Constructing a layout does not write to disk.  Metadata writes are atomic.
`ensure_run_directories()` creates only the run root (and optionally
`artifacts/`), never `shards/` or `repetitions/`.

## GQA shards

A shard is an execution detail of one logical run, not another experiment
dimension.  At present it is used only by GQA multi-GPU launchers:

```text
<run_id>/
  shards/
    shard_000/
      run.config.json
      run.status.json
      results.jsonl
      results.summary.json
      verifier_events.jsonl
      run.log
    shard_001/
      ...
  results.jsonl             # deterministically merged from shards
  results.summary.json      # recomputed from the merged records
```

`NUM_SHARDS=1` writes evaluator output directly in `<run_id>/` and does not
create `shards/`; no merge step is needed. Only `NUM_SHARDS>1` creates
`shards/shard_NNN/`, followed by deterministic aggregation into the root
`results.jsonl` and recomputation of the root `results.summary.json`. When
`NUM_SHARDS` is not set, the GQA launcher uses the number of selected GPUs.
During a multi-partition run, progress/status lives in each shard; the root
config and terminal status become authoritative when aggregation completes.

The launcher resolves the canonical run root once, then creates exact shard
paths below `layout.shards_dir`.  A shard evaluator adapts the path without
changing it:

```python
from verifier.run_paths import create_exact_output_layout

shard = create_exact_output_layout(
    dataset='gqa',
    split='testdev_12578',
    study='baseline',
    method='volcano_7b',
    setting='default',
    run_id=logical_run_id,
    output=args.output,
)
```

It must not call `create_run_layout(output=<shard-file>)`: that branch adds the
legacy timestamp/run-id level and is only for evaluators that previously
called `resolve_run_output`. The exact adapter requires the shared logical
`run_id`, writes config/status next to the resolved result, and inserts no path
component. It is also used by older single-process CLIs whose historical
`--output` contract named the exact file; those direct calls may omit the run
ID and receive a metadata-only timestamp without changing the requested file
path. VStar and single-process runs should not create an empty `shards/`
directory.

Resuming a GQA run reuses the same `run_id` and shard count.  A changed split,
method, setting, or shard assignment is a new run.  Only the merged root result
should be referenced by reports or downstream experiments.

The launcher records `artifacts/partition_plan.txt` before starting workers.
Reusing a run root with a different manifest range, partition count, model, or
primary setting fails instead of silently merging stale shard files.
Aggregation also requires exactly `shard_000 ... shard_NNN` and exact
`sample_index` coverage of the requested interval; a missing, stale, duplicate,
or out-of-range record marks the root run as failed before merged results are
written.

## Repetitions

`repetitions/` is for statistically independent runs such as matched-random
seeds.  It is not interchangeable with shards:

```text
<run_id>/
  repetitions/
    repeat_01/
    repeat_02/
  results.summary.json       # aggregate across repetition summaries
```

Each repetition evaluates the full selected population.  Shards partition one
population and are merged; repetitions repeat it and are aggregated with mean,
standard deviation, and range. Their duplicate per-sample JSONLs are not
concatenated into a root `results.jsonl`; the root summary records every source
repetition and explicitly marks `sample_jsonl_merged: false`.

## Non-run data and reports

Stable source data remains separate from generated runs:

```text
output/<dataset>/annotations/
output/<dataset>/reports/tables/<comparison_id>/
output/<dataset>/reports/figures/<comparison_id>/
output/<dataset>/reports/renders/<source_run_id>/
output/verifier_benchmark/cache/<benchmark_split>/<backend>/<cache_key>/
```

Do not move annotation manifests merely to make them fit the run hierarchy.
Reports should identify their source run IDs and never mutate source results.
The verifier benchmark cache is parameter-addressed and reusable across run
IDs; each consuming threshold-search run stores a reference under its own
`artifacts/` instead of copying the cached inference.

## Explicit-output compatibility

There are two deliberately separate compatibility contracts. Evaluators that
already used `resolve_run_output` pass an explicit output to
`create_run_layout(output=...)` and retain their timestamped placement:

```text
requested: output/vstar/online_oracle/results.jsonl
resolved:  output/vstar/online_oracle/<run_id>/results.jsonl
```

Evaluators whose older CLI wrote the exact requested filename use
`create_exact_output_layout`; this includes runner-resolved GQA paths, the
original VStar counterfactual/oracle entry points, and the verifier benchmark
CLI. Omitting `output` selects the canonical hierarchy in either case. These
compatibility escapes are not recommended layouts for new runners.

See [LEGACY_INDEX.md](LEGACY_INDEX.md) for the preserved historical trees and
their canonical study mapping.
