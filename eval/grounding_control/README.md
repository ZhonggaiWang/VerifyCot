# Grounding-control evaluations

## VStar Grounder accuracy

`vstar/evaluate_grounder_accuracy.py` measures a Grounder independently from
the verifier, router, VoCoT candidate, and final QA task.  Each request contains
only the immutable source image and one canonical VStar `target_object`.

The `full_238` annotation file expands to 292 object--box requests.  Predictions
are evaluated against GT in original-image continuous pixel `xyxy` coordinates.
An unavailable/parse-failed prediction remains a completed measurement with
IoU zero; infrastructure failures terminate the run.

Run the default Qwen2.5-VL-7B configuration:

```bash
QWEN_GPU=7 RUN_ID=qwen7b_full_v1 \
  scripts/run_vstar_qwen_grounder_accuracy.sh
```

Run a three-target smoke test:

```bash
QWEN_GPU=7 MAX_TARGETS=3 RUN_ID=qwen7b_smoke \
  scripts/run_vstar_qwen_grounder_accuracy.sh
```

Canonical output:

```text
output/vstar/runs/full_238/grounder_accuracy/
  qwen25_vl_7b/compact_json_v1/<run_id>/
    run.config.json
    run.status.json
    results.jsonl
    results.summary.json
```

The evaluator also accepts `--backend grounding_dino`; both workers implement
the same `GrounderBackend` and `vocot_grounder_output_v1` boundary, allowing
later backends to reuse exactly the same tasks and metrics.

## VStar routing: Oracle verifier + Qwen7B Grounder

`vstar/evaluate_oracle_verifier_qwen_grounder.py` measures the correction
ability of a real Grounder while holding verifier errors out of the experiment.
It uses the padding-fixed `full_238` records as the paired baseline and GT
source.  At every natural VoCoT coordinate boundary:

1. A conservative resolver uniquely matches the local object reference to a
   VStar target when possible.
2. The Oracle binary verifier accepts the candidate at IoU `>= 0.5` and
   rejects it below `0.5`.
3. Only a rejection calls the persistent Qwen2.5-VL-7B Grounder.  The Grounder
   receives the clean source image and generated object reference; it never
   receives the candidate box, GT box, verifier label, or task question.
4. The selected box is committed through normal Volcano REFbind and later CoT
   remains freely generated.  Unmatched references are accepted fail-open.

Run all 238 questions with Volcano on physical GPU 6 and Qwen on GPU 7:

```bash
GENERATOR_GPU=6 QWEN_GPU=7 \
RUN_ID=oracle_verifier_qwen7b_iou05_v1 \
  scripts/run_vstar_oracle_verifier_qwen_grounder.sh
```

Run one end-to-end sample that exercises the Grounder path:

```bash
GENERATOR_GPU=6 QWEN_GPU=7 SAMPLE_ID=main:9 \
RUN_ID=smoke_main9 \
  scripts/run_vstar_oracle_verifier_qwen_grounder.sh
```

Canonical output:

```text
output/vstar/runs/full_238/routing/
  oracle_verifier__qwen25_vl_7b_grounder/gt_iou_0p5__12mp/<run_id>/
    run.config.json
    run.status.json
    verifier_events.jsonl
    results.jsonl
    results.summary.json
```

The summary reports paired baseline/router QA accuracy and transitions, exact
McNemar p-value, Grounder call rate, Qwen committed mIoU and recall, per-call
IoU gain, successful-error correction counts, complete-target-coverage subset,
and VStar category breakdowns.  `--run-id` resumes successful samples by
default; set `NO_RESUME=1` in the launcher to restart that run ID.
