# Pre-commit grounding control system

The package owns the complete pre-commit control path: candidate verification,
confidence routing, expert correction, and clean coordinate commitment.  The
verifier is one role inside this system rather than the name of the system.

The current implementation is split by responsibility:

- `core/coordinate_rollout.py`: policy-neutral coordinate boundary and clean
  replay mechanics.
- `core/precommit_controller.py`: binary verification, Grounder dispatch, and
  clean commit.
- `core/alignment_policy.py` and `core/calibration.py`: binary score policy
  and optional calibration.
- `contracts/`: stable binary verifier and Grounder interfaces.
- `models/`: reusable Qwen and Grounding DINO inference capabilities, without
  controller-role policy.
- `verifiers/`: binary verifier-role implementations, including the binary
  oracle used for upper bounds.
- `experts/grounders/`: routed correction experts; the oracle implementation
  returns a conservatively matched GT box.
- `four_way/`: archived action verifier, Grounder-only router, and worker stack.
- `oracle_targets.py`: shared reference-to-GT matcher for oracle components.
- `transport/`: model-agnostic persistent JSONL protocol and clients.
- `workers/`: binary verifier and standalone `dino_grounder` / `qwen_grounder`
  expert worker entry points; action compositions live under
  `four_way/workers/`.
- `legacy/repair_controller.py`: archived prompt-repair, one-shot corruption,
  and sandbox REFbind ablations retained for exact reproduction.
- `legacy/oracle_backends/`: stored and single-candidate oracle lookups used
  only by those archived repair experiments.

The controller always stops before a candidate coordinate enters
REFbind. It then commits either the accepted candidate or the Grounder's
replacement; the committed coordinate always follows
Volcano's normal REFbind path. There is no active sandbox REFbind mode.

The primary routing interface is binary alignment scoring with explicit
accept, reject, and uncertain bands. The retained four-way interface is an
explicit diagnostic/appendix path under `grounding_control.four_way`.

Grounder backends can be compared independently of routing with
`eval/grounding_control/vstar/evaluate_grounder_accuracy.py`.  Its production
Qwen2.5-VL-7B launcher is
`scripts/run_vstar_qwen_grounder_accuracy.sh`; the benchmark sends only a
clean image and canonical object reference and scores original-image boxes.

The complementary end-to-end VStar routing evaluator is
`eval/grounding_control/vstar/evaluate_oracle_verifier_qwen_grounder.py`.
It verifies every natural coordinate with a binary GT-IoU oracle and calls the
real Qwen2.5-VL-7B Grounder only for rejected coordinates.  Use
`scripts/run_vstar_oracle_verifier_qwen_grounder.sh`; detailed semantics and
output fields are documented in `eval/grounding_control/README.md`.

## Qwen2.5-VL verifier backend

Qwen keeps two zero-shot judgment protocols in separate namespaces:

- `verifiers/qwen25_vl/` implements `binary_alignment`, which returns whether
  the candidate region supports the object reference.
- `four_way/verifiers/qwen25_vl/` implements `routing_four_way`, which predicts
  `no_action`, `relocate`, `expand`, or `tighten`.

Both consume the source PIL image, the local object reference, and the
uncommitted VoCoT candidate box. They deliberately do not consume the original
task question: verification is a local object--region alignment decision. The
binary protocol returns:

```json
{"aligned":true,"confidence":0.9}
```

The direct four-way protocol returns:

```json
{"status":"relocate","confidence":0.9}
```

Confidence is the model's self-reported value from 0.0 to 1.0.

The candidate box is already normalized on VoCoT's center-padded square.
The renderer reproduces `VoCoT_InputProcessor.expand2square_fn` and draws the
box directly on that square; it never applies the original-image-to-padding
conversion a second time.

Qwen imports and model loading are lazy. The original Volcano environment can
therefore import this backend. Local benchmark evaluation uses
`LocalQwen25VLRunner` in the Qwen-compatible environment; split-environment
Archived four-way VoCoT inference uses
`grounding_control.four_way.RemoteActionVerifierBackend` with its persistent
JSONL worker, so the Volcano process does not import or load Qwen itself.

## Controlled GQA verifier benchmark

`benchmarks/gqa_controlled/` adapts the generated GQA JSONL records to the
same Qwen inputs used by the online backend. Depending on the selected
protocol, it sends the candidate crop, the complete scene with the candidate
marked in red, or both. The adapter reads only the source image, object
reference, and candidate pixel box before inference. Target boxes,
construction metadata, verdicts, and reasons are retained only for
post-inference scoring.

The adapter converts the original-image pixel box exactly once into VoCoT's
normalized center-padded-square coordinate system. Existing benchmark renders
are intentionally ignored by this production-faithful path; the Qwen backend
re-renders its own red candidate rectangle.

The local runner imposes no project-level pixel cap by default. For each
request it raises Qwen's processor limit to the factor-rounded source-image
area, so the available source resolution is retained. `--max-pixels` remains
available only for an explicit resolution ablation. A pixel limit cannot
compensate for a GPU already occupied by another process; benchmark runs
should use an otherwise free device.

Qwen rejects images whose width or height does not exceed its 28px merged
patch factor. Candidate crops with a short side below 56px are therefore
upscaled proportionally to a 56px short side before model input. The raw
candidate field of view is unchanged. Result metadata records both
`candidate_crop_size` and `model_crop_size`; use `--crop-min-side` to change
the threshold for an ablation.

The binary benchmark exposes an explicit image-context ablation:
`--binary-image-mode crop_only` sends only the candidate crop,
`--binary-image-mode bbox_image_only` sends only the complete scene with its
red candidate box, and `--binary-image-mode marked_plus_crop` sends that marked
scene followed by the identical crop. Each mode uses an explicit
mode-appropriate prompt while retaining the same labels and decoding settings.

The direct action task is selected with `--task-mode routing_four_way`. The
controlled benchmark retains its source construction labels in the result
JSONL, but maps them to routing actions for scoring as follows:

- `aligned -> no_action`
- `wrong_object / unsupported -> relocate`
- `partial_coverage -> expand`
- `ambiguous -> tighten`

Select its image input with `--routing-image-mode`; the default is
`bbox_image_only`.

An alternative geometry-based route is available through
`--task-mode routing_grounding_geometry`. Instead of asking Qwen to select an
action directly, it asks Qwen to locate exactly one instance of the object
reference and return one absolute `xyxy` box on the exact smart-resized image.
The candidate coordinates are hidden from the localization prompt. A
model-independent router then compares the candidate and generated reference
box:

- IoU at least `--grounding-accept-iou` (default `0.5`) -> `no_action`
- candidate coverage at least `--grounding-containment` (default `0.7`) ->
  `expand`
- generated-reference coverage at least the containment threshold ->
  `tighten`
- all remaining low-IoU relations -> `relocate`

The `expand` and `tighten` rules apply only below the acceptance IoU and
require asymmetric containment. This avoids inferring a size correction from
box area alone. Use `--grounding-image-mode raw_image` to hide the candidate
entirely, or `bbox_image` to show it as a red visual hint. Parsing, geometry,
prompt construction, and model execution remain separate modules.

The grounding parser tolerates at most one generated pixel beyond each image
edge by default. It clips only that bounded excursion and records the raw box,
usable clipped box, affected sides, and tolerance in result metadata. Larger
excursions, empty boxes, and multiple boxes remain failures. Set
`--grounding-boundary-tolerance 0` for strict bounds or another non-negative
value for an explicit ablation.

Two localization prompt protocols are retained for controlled comparison:
`--grounding-prompt-protocol compact_json_v1` reproduces the original concise
prompt, while `single_object_json_v2` explicitly forbids top-level lists,
multiple boxes, labels inside `bbox_2d`, Markdown, and explanations. The
concise protocol remains the default because the evaluator's default model is
7B. On the current GQA dev split, strict v2 substantially improves 3B parsing
and routing, but compact v1 retains better 7B routing quality; select the
protocol explicitly when reporting an ablation.

### Benchmark output layout

New benchmark runs follow the same canonical contract as model evaluations:

```text
output/verifier_benchmark/runs/<benchmark_split>/<study>/<method>/<setting>/<run_id>/
```

For the controlled V1 data, the exact evaluated population is
`gqa_controlled_v1_dev`, `gqa_controlled_v1_test`, or
`gqa_controlled_v1_all`. Typical studies are `binary`, `four_way`, and
`threshold_search`; the backend/model belongs in `method`, while the
prompt/image protocol or primary threshold belongs in `setting`. Full
parameters remain in run metadata rather than the path.

The evaluator resolves this path automatically when `--output` is omitted and
writes `run.config.json` plus `run.status.json`. Explicit `--output` keeps the
old exact-filename behavior for archived commands and custom launchers. It
does not add a timestamp directory. Existing benchmark outputs and the stable
controlled V1 input remain at their historical paths and are not moved.

Run the direct four-way verifier with:

```bash
RUN_ID=20260801_153000__qwen7b_direct
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  grounding_control.benchmarks.gqa_controlled.evaluator \
  --model-path weights/Qwen2.5-VL-7B-Instruct \
  --task-mode routing_four_way \
  --routing-image-mode bbox_image_only \
  --split test \
  --run-id "$RUN_ID" \
  --verbose
```

Run the geometry-based variant with:

```bash
RUN_ID=20260801_153000__qwen7b_geometry
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  grounding_control.benchmarks.gqa_controlled.evaluator \
  --model-path weights/Qwen2.5-VL-7B-Instruct \
  --task-mode routing_grounding_geometry \
  --grounding-image-mode raw_image \
  --grounding-accept-iou 0.5 \
  --grounding-containment 0.7 \
  --grounding-boundary-tolerance 1 \
  --grounding-prompt-protocol single_object_json_v2 \
  --split dev \
  --run-id "$RUN_ID" \
  --verbose
```

The same geometry task can use Grounding DINO as an independent reference
grounder. In this mode the detector sees only the clean original image and
object reference. Its highest-score detection is post-processed back to
absolute `xyxy` coordinates on the original image and compared directly with
the candidate; neither Qwen smart resize nor VoCoT square padding is involved.
No detection is recorded as an end-to-end localization failure rather than
being guessed as `relocate`.

Grounding DINO requires the dedicated `qwen25` environment (Transformers 4.49)
and a local checkpoint. In the current top-1 geometry protocol,
`box_threshold` controls whether the highest-score localization is retained,
while `text_threshold` changes only its decoded phrase label and does not
affect routing. Select the box threshold on `dev`, keep the text threshold
fixed, then freeze both before evaluating `test`.

The threshold tuner runs DINO only once at the lowest requested threshold and
replays every larger threshold offline. By default it selects the best dev
Macro-F1:

```bash
RUN_ID=20260801_153000__macro_f1 GPU_ID=7 \
  scripts/tune_grounding_dino_dev_thresholds.sh
```

Set `BOX_THRESHOLDS`, `TEXT_THRESHOLD`, or `SELECTION_METRIC` to change the
search. Compatible low-threshold inference is reused across run IDs from a
parameter-addressed cache under `output/verifier_benchmark/cache/`; set
`FORCE_INFERENCE=1` to regenerate it or `CACHE_DIR` to choose an exact shared
cache. Canonical runs write threshold rows to `results.jsonl`, the full report
to `results.summary.json`, and the best config, CSV table, and cache-reference
record below `artifacts/`. Set `OUTPUT_DIR` only to reproduce the old exact
directory layout, including its local `inference_cache/`.

One direct evaluator invocation is:

```bash
RUN_ID=20260801_153000__dino_geometry
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  grounding_control.benchmarks.gqa_controlled.evaluator \
  --model-path weights/grounding-dino-base \
  --task-mode routing_grounding_geometry \
  --geometry-backend grounding_dino \
  --grounding-image-mode raw_image \
  --grounding-accept-iou 0.5 \
  --grounding-containment 0.7 \
  --dino-box-threshold 0.30 \
  --dino-text-threshold 0.25 \
  --split dev \
  --run-id "$RUN_ID" \
  --verbose
```

Run a small development smoke test in the dedicated Qwen environment:

```bash
RUN_ID=20260801_153000__smoke
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  grounding_control.benchmarks.gqa_controlled.evaluator \
  --split dev \
  --limit 20 \
  --run-id "$RUN_ID" \
  --verbose
```

After freezing the prompt, run the held-out test split by removing `--limit`
and changing `--split test`. The evaluator writes one resumable JSONL plus
`results.summary.json` containing binary or four-way accuracy, macro-F1 where
applicable, per-class metrics, confusion matrix, parse success, and
self-reported confidence summaries.

## Legacy oracle repair input

Use JSONL (or a JSON list) with one record for every candidate that may be
checked. The key is exactly `(sample_id, grounding_step, attempt_index)`.

```json
{"sample_id":"sample_001","grounding_step":2,"attempt_index":0,"object_reference":"the bar for series B in 2019","candidate_bbox":[0.12,0.30,0.24,0.78],"verifier_output":{"verdict":"misaligned","reason":"wrong_object","confidence":1.0}}
{"sample_id":"sample_001","grounding_step":2,"attempt_index":1,"candidate_bbox":[0.34,0.30,0.46,0.78],"verifier_output":{"verdict":"aligned","reason":"none","confidence":1.0}}
```

`candidate_bbox` is optional, but recommended: when present it prevents an
oracle decision prepared for one generated coordinate from being applied to a
different coordinate. Missing records and mismatches become `uncertain`; they
never accept a box.

## Running the archived prompt-repair experiment

```python
from model.load_model import infer

response, metadata = infer(
    model, preprocessor, image, question, cot=True,
    verifier_oracle_file='path/to/oracle.jsonl',
    verifier_sample_id='sample_001',
    verifier_repair_mode='typed_feedback',  # blind_retry | binary_feedback | typed_feedback
    verifier_accept_confidence=0.8,
    verifier_max_retries=2,
    verifier_on_failure='skip_grounding_and_continue',  # or abort_sample
    verifier_log_path='path/to/run/verifier_events.jsonl',
    return_metadata=True,
)
```

This archived API takes an explicit log path for exact reproduction. Place new
logs inside the owning canonical run; historical repair logs are not moved.

Calling `infer()` without `verifier_oracle_file` follows the original code
path. `verifier_infer()` is also exported from `model.load_model` when a
dedicated entry point is clearer.

## Clean commit semantics

For every coordinate candidate, the controller stops at the first generated
`</coor>` before the next model forward, so `generate_box()` cannot inject its
feature. It verifies the candidate, then starts a new generation from the
original prompt and force-replays only accepted persistent tokens. The native
VoCoT REFbind call runs while replaying accepted closing tags. Thus rejected
coordinates and repair feedback are absent from the subsequent persistent CoT.

Temporary feedback includes a literal rejected `<coor>...</coor>` and follows
the original Volcano REFbind path inside the sandbox. Thus its visual feature
can influence replacement generation. The repair call ends at the first
replacement `</coor>`; that candidate is still verified before commit, and the
entire sandbox is discarded before clean replay of an accepted replacement.

Every binary/typed repair prompt ends in the exact training-style boundary
`... re-reason as {object_reference}<coor>`: no quote, punctuation, or other
text token separates the reference from the opening coordinate tag.

The first version deliberately uses no KV-cache snapshot or reuse. It uses
full clean replay/prefill, which is slower but makes rollback unambiguous.

## Event log example

```json
{"sample_id":"sample_001","grounding_step":2,"object_reference":"the bar for series B in 2019","h_t_ends_before_coor":true,"initial_bbox":[0.12,0.3,0.24,0.78],"initial_verdict":"misaligned","initial_reason":"wrong_object","repair_mode":"typed_feedback","repair_attempts":[{"attempt_index":1,"rejected_bbox_in_prompt":[0.12,0.3,0.24,0.78],"generated_bbox":[0.34,0.3,0.46,0.78],"verdict":"aligned","reason":"none","confidence":1.0,"missing_oracle_record":false,"oracle_candidate_mismatch":false}],"committed_bbox":[0.34,0.3,0.46,0.78],"repair_success":true,"rejected_coor_in_persistent_context":false,"feedback_in_persistent_context":false,"visual_feature_injected":true,"latency_ms":42.0}
```

After all repair attempts fail, the controller replays H_t and masks `<coor>`
for exactly the first free decoder choice. It then releases the generator to
finish its revised CoT. This removes the failed object's immediate grounding
decision without committing its rejected coordinates; later coordinates remain
available to the free revised trajectory.
