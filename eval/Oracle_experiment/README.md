# Grounding intervention evaluations

This directory is organised by dataset:

- `vstar/`: VStar counterfactual, online-oracle, and box-audit scripts.
- `gqa/`: GQA-val subset construction plus counterfactual and online-oracle evaluations.

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
  --output output/gqa/counterfactual/random_box/random_seed2026/results.jsonl \
  --perturb-mode random_box --perturb-position random
```

Use `--perturb-position first` or `last` for the fixed-position settings;
use `--perturb-mode remove_grounding` to omit the selected coordinate and its
REFbind feature altogether.  Output automatically resumes by `sample_index`;
pass `--no-resume` to start a fresh run.

For a multi-GPU run, the scheduler partitions the manifest into non-overlapping
contiguous `sample_index` shards, waits for an allowed idle GPU for each shard,
and merges all shard results automatically. `GPU_IDS` uses physical
`nvidia-smi` indices.

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
  --output output/gqa/online_oracle/strict_name_gt/results.jsonl
```

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

The suite keeps stages separate under
`output/gqa/full_intervention_suite/<timestamp>/`; within a stage, all selected
cards run independent shards in parallel.

Each `results.summary.json` contains paired baseline/intervention accuracy,
answer-change counts, correctness transitions, and structural-question-type
breakdowns.
