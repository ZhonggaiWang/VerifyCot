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
expand    -> OracleBoxRefinerBackend
tighten   -> OracleBoxRefinerBackend
```

The oracle experts use the latest-unique-longest explicit alias policy. If the
current reference cannot be matched uniquely to an annotated object, the
expert is unavailable and the controller fails open by committing the model
box. This unmatched case is recorded in `missing_expert_error`.

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
