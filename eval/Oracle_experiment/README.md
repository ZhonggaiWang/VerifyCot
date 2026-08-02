# Grounding intervention evaluations

This directory is organised by dataset:

- `vstar/`: VStar counterfactual, online-oracle, and box-audit scripts.
- `gqa/`: GQA-val subset construction plus counterfactual and online-oracle evaluations.

## Output contract

New experiment results use one directory per logical run:

```text
output/<dataset>/runs/<split>/<study>/<method>/<setting>/<run_id>/
```

The run root contains the authoritative `results.jsonl`,
`results.summary.json`, `run.config.json`, and `run.status.json`; verifier runs
may additionally write `verifier_events.jsonl`. Annotation manifests remain
stable inputs under `output/<dataset>/annotations/` and do not belong below
`runs/`.

Where an evaluator supports it, omit `--output` and use `--run-id` (plus
`--run-split` for a non-default split) to select this layout. An explicit
`--output` is retained as a compatibility interface for archived commands and
launcher-resolved exact paths; it is not a second recommended hierarchy.
Historical results remain at their recorded paths and are not moved. See
`output/README.md` and `output/LEGACY_INDEX.md` for the complete contract and
legacy mapping.

## GQA oracle subset

The GQA scripts use the official `val_balanced_questions.json` and
`val_sceneGraphs.json`.  The manifest is a fixed, seed-controlled stratified
subset: each sample has an existing image, one to three question/semantic-program
object targets, and a valid GT scene-graph box.

Build (or reproduce) the 1,000-sample manifest:

```bash
python eval/Oracle_experiment/gqa/build_oracle_subset.py \
  --questions-path /data/zhonggai/GQA/val_balanced_questions.json \
  --scene-graphs-path /data/zhonggai/GQA/val_sceneGraphs.json \
  --image-dir /data/zhonggai/GQA/images \
  --output output/gqa/annotations/oracle_val_1000/manifest.jsonl \
  --count 1000 --seed 20260724
```

## GQA counterfactual experiment

The normal rollout produces a CoT.  The corresponding counterfactual rollout
replays its prefix through one selected coordinate, then either inserts a random
wrong box (`random_box`) or removes that grounding event (`remove_grounding`).
The remaining CoT and final short answer are freely generated.  GQA is an
open-ended-answer dataset, so final predictions are scored by the repository's
normalised exact-answer comparison, not option likelihood.

```bash
CUDA_VISIBLE_DEVICES=0 python eval/Oracle_experiment/gqa/evaluate_counterfactual.py \
  --manifest-path output/gqa/annotations/oracle_val_1000/manifest.jsonl \
  --perturb-mode random_box --perturb-position random \
  --run-split val_1000_dev \
  --run-id 20260801_153000__random_box
```

This example writes to
`output/gqa/runs/val_1000_dev/counterfactual/random_box/random/20260801_153000__random_box/`.

Use `--perturb-position first` or `last` for the fixed-position settings;
use `--perturb-mode remove_grounding` to omit the selected coordinate and its
REFbind feature altogether.  Output automatically resumes by `sample_index`;
pass `--no-resume` to start a fresh run.

The GQA launcher handles one logical run in either direct or sharded mode.
With `NUM_SHARDS=1`, it writes `results.jsonl`, `results.summary.json`, and the
run metadata directly in the canonical run root and does not create
`shards/`. With `NUM_SHARDS>1`, it partitions the manifest into non-overlapping
contiguous `sample_index` ranges, writes each partial run under
`shards/shard_NNN/`, then deterministically merges and re-summarises the
authoritative root result. `GPU_IDS` uses physical `nvidia-smi` indices. If
`NUM_SHARDS` is omitted, it defaults to the number of selected GPUs.
Before merging, the launcher verifies both the exact shard set and continuous
coverage of the requested `sample_index` interval, so an incomplete run cannot
silently become a root summary.

Single-partition launcher example:

```bash
GPU_IDS=0 NUM_SHARDS=1 \
PERTURB_MODE=random_box PERTURB_POSITION=random \
scripts/run_gqa_counterfactual.sh
```

Multi-partition launcher example:

```bash
GPU_IDS=0,1,2 NUM_SHARDS=3 \
PERTURB_MODE=random_box PERTURB_POSITION=random \
scripts/run_gqa_counterfactual.sh
```

Set `PERTURB_POSITION=first` or `last`, or use `PERTURB_INDEX=2` for a fixed
1-based coordinate index. `PERTURB_MODE=remove_grounding` enables the
no-grounding intervention. Use `EVAL_MAX_SAMPLES=100` for a pilot run.

## GQA online oracle experiment

During CoT generation, each generated coordinate is conservatively matched
against the GT target names from the manifest.  A unique explicit name match
replaces both the emitted coordinate text and the associated REFbind region
feature with the GT box.  Ambiguous or unmatched references remain model
generated.

```bash
CUDA_VISIBLE_DEVICES=0 python eval/Oracle_experiment/gqa/evaluate_online_oracle.py \
  --manifest-path output/gqa/annotations/oracle_val_1000/manifest.jsonl \
  --run-split val_1000_dev \
  --run-id 20260801_153000__always_gt
```

This writes to
`output/gqa/runs/val_1000_dev/oracle/always_gt/default/20260801_153000__always_gt/`.

The multi-GPU equivalent is:

```bash
GPU_IDS=0,1,2 NUM_SHARDS=3 scripts/run_gqa_online_oracle.sh
```

To run the complete GQA schedule sequentially—remove-grounding random/first/
last, random-box random/first/last, then online oracle—use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NUM_SHARDS=3 \
scripts/run_gqa_full_intervention_suite.sh
```

By default the suite keeps each stage in its canonical study/method/setting
directory under `output/gqa/runs/` and reuses one `SUITE_TAG` as the `run_id`
across those separate stage directories. Within a stage, selected cards run
data-parallel shards only when `NUM_SHARDS>1`. Setting `SUITE_ROOT` (or the
legacy `OUTPUT_ROOT` alias) explicitly retains the older custom nested-suite
placement for compatibility.

Each `results.summary.json` contains paired baseline/intervention accuracy,
answer-change counts, correctness transitions, and structural-question-type
breakdowns.
