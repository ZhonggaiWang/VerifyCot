# Remote DINO verifier with oracle experts

This upper-bound experiment runs Volcano and Grounding DINO in separate
processes:

```text
physical GPU 6: Volcano / VoCoT controller
physical GPU 7: persistent Grounding DINO geometry verifier
```

Every generated coordinate is stopped before REFbind. DINO localizes the
current reference without seeing the candidate. Deterministic geometry maps
the candidate and DINO box to:

```text
no_action -> keep the model box
relocate  -> OracleGrounderBackend
expand    -> OracleGrounderBackend
tighten   -> OracleGrounderBackend
```

The four-way labels are retained only for archived diagnostics. Every
non-accept label now has the same operational meaning: reject the candidate
and ask the Grounder to localize the referenced object independently.

The oracle experts use the latest-unique-longest explicit alias policy. If the
current reference cannot be matched uniquely to an annotated object, the
expert is unavailable and the controller fails open by committing the model
box. This unmatched case is recorded in `missing_expert_error`.

An expert result is formatted and round-trip validated locally, then committed
on the next clean replay. It does not trigger a separate, fully forced Volcano
generation pass before that replay. The event fields
`expert_coordinate_commit_mode` and
`expert_coordinate_extra_model_forward` make this runtime choice auditable.

## Recommended launcher

Use the repository launcher so the two Python environments and physical GPU
indices stay explicit. It writes through the evaluator's canonical run layout:

```text
output/vstar/runs/full_238/routing/
  dino_geometry__oracle_experts/iou_<threshold>/<run_id>/
```

First inspect one sample without starting either model:

```bash
GENERATOR_GPU=6 \
DINO_GPU=7 \
SAMPLE_ID=main:9 \
RUN_ID=smoke_main_9 \
DRY_RUN=1 \
  scripts/run_vstar_dino_geometry_oracle_experts.sh
```

Then remove `DRY_RUN=1` to execute it. A complete run at the exploratory
IoU setting `0.6` is:

```bash
GENERATOR_GPU=6 \
DINO_GPU=7 \
GEOMETRY_ACCEPT_IOU=0.6 \
RUN_ID=full_238_iou06_v1 \
  scripts/run_vstar_dino_geometry_oracle_experts.sh
```

The launcher defaults to the evaluator's reproducible `0.4` accept-IoU
setting; always set `GEOMETRY_ACCEPT_IOU` explicitly for a reported run.
Other useful controls are:

- `DINO_BOX_THRESHOLD`, `DINO_TEXT_THRESHOLD`, `GEOMETRY_CONTAINMENT`
- `SAMPLE_ID`, or `START_INDEX` plus `MAX_SAMPLES`
- `FAILURE_POLICY=fail_fast|fail_open` (default `fail_fast`)
- `MISSING_EXPERT_POLICY=fail_open|error` (default `fail_open`)
- `RUN_SPLIT` (default `full_238`)
- `NO_RESUME=1`, `VERBOSE=0|1`, and `DRY_RUN=1`
- `VOCOT_PYTHON` and `DINO_PYTHON` for exact environment selection
- `MODEL_PATH`, `DINO_MODEL_PATH`, `BASELINE_RESULTS`, and `IMAGE_DIR`

`FAILURE_POLICY` applies to failed DINO worker/sample requests. Oracle expert
unavailability from an unmatchable object reference is controlled separately
by `MISSING_EXPERT_POLICY`; the default keeps the model candidate and records
`missing_expert_error`.

## Direct evaluator invocation

Run one VStar sample first:

```bash
CUDA_VISIBLE_DEVICES=6 \
/home/zhonggai/miniconda3/envs/vocot/bin/python -u \
  eval/Oracle_experiment/vstar/evaluate_dino_geometry_oracle_experts.py \
  --dino-gpu 7 \
  --dino-python /home/zhonggai/miniconda3/envs/qwen25/bin/python \
  --dino-model-path /data/zhonggai/models/grounding-dino-base \
  --sample-id main:9 \
  --run-id smoke_main_9 \
  --verbose
```

Run the complete padding-fixed VStar set by removing `--sample-id`:

```bash
CUDA_VISIBLE_DEVICES=6 \
/home/zhonggai/miniconda3/envs/vocot/bin/python -u \
  eval/Oracle_experiment/vstar/evaluate_dino_geometry_oracle_experts.py \
  --dino-gpu 7 \
  --dino-python /home/zhonggai/miniconda3/envs/qwen25/bin/python \
  --dino-model-path /data/zhonggai/models/grounding-dino-base \
  --run-id full_238_v1 \
  --verbose
```

The script overrides `CUDA_VISIBLE_DEVICES` only for the DINO subprocess, so
the worker sees physical GPU 7 as logical `cuda:0`. The generator process
continues to see only physical GPU 6.

`--verifier-confidence-threshold` defaults to `0.0`: DINO detector scores are
not calibrated four-action probabilities. Use a nonzero threshold only after
calibration. Worker failures are strict by default; `--worker-fail-open`
converts them to verifier abstentions for robustness experiments.

The dedicated experiment and DINO worker use `0.4` as the default geometry
accept-IoU threshold. The older generic/offline geometry evaluators retain
their historical `0.5` defaults so previous results remain reproducible.
