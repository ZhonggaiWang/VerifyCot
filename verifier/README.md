# Object--coordinate verifier and router

The active implementation is split into three layers:

- `routing_controller.py`: model-agnostic coordinate boundary, verification,
  expert-grounder routing, and clean commit.
- `backend.py` and `backends/`: verifier/grounder interfaces and concrete
  backends. `backends/oracle/` is restricted to upper-bound experiments.
- `legacy/repair_controller.py`: archived prompt-repair, one-shot corruption,
  and sandbox REFbind ablations retained for exact reproduction.
- `legacy/oracle_backends/`: stored and single-candidate oracle lookups used
  only by those archived repair experiments. Root-level modules with the old
  names are compatibility re-exports.

The active controller always stops before a candidate coordinate enters
REFbind. It then commits either the accepted candidate or the grounding
backend's replacement; the committed coordinate always follows Volcano's
normal REFbind path. There is no active sandbox REFbind mode.

## Qwen2.5-VL verifier backend

`backends/qwen25_vl/` implements a zero-shot five-way verifier. It consumes
the source PIL image, the local object reference, and the uncommitted VoCoT
candidate box. It deliberately does not consume the original task question:
verification is a local object--region alignment decision.
The model is explicitly prompted to return JSON with:

```json
{"status":"aligned","confidence":0.9}
```

The value must be exactly one of `aligned`, `wrong_object`,
`partial_coverage`, `ambiguous`, or `unsupported`. Confidence is the model's
self-reported value from 0.0 to 1.0.

`unsupported` means that the candidate region falls on background or contains
no visual evidence for the referenced object. It is represented internally as
`misaligned / unsupported`, alongside `misaligned / wrong_object`; it does not
mean that the referenced object is absent from the entire image.

The candidate box is already normalized on VoCoT's center-padded square.
The renderer reproduces `VoCoT_InputProcessor.expand2square_fn` and draws the
box directly on that square; it never applies the original-image-to-padding
conversion a second time.

Qwen imports and model loading are lazy. The original Volcano environment can
therefore import this backend, while `LocalQwen25VLRunner` must execute in a
separate environment with Qwen2.5-VL support. A future process/RPC runner can
implement the same `Qwen25VLRunner.generate(messages)` interface without
changing the backend.

## Controlled GQA verifier benchmark

`benchmarks/gqa_controlled/` adapts the generated GQA JSONL records to the
same two-image Qwen input used by the online backend: one full scene with the
candidate marked in red, followed by a border-free crop from inside that box.
The unmarked full image is not sent separately. The adapter reads only the
source image, object reference, and candidate pixel box before inference.
Target boxes, construction metadata, verdicts, and reasons are retained only
for post-inference scoring.

The adapter converts the original-image pixel box exactly once into VoCoT's
normalized center-padded-square coordinate system. Existing benchmark renders
are intentionally ignored by this production-faithful path; the Qwen backend
re-renders its own red candidate rectangle.

The local runner caps each input image at `512 * 28 * 28 = 401408` pixels by
default, or about 512 merged Qwen visual tokens per image. This bounds the
two-image verifier's activation peak. Override it with `--max-pixels` when a
different accuracy/memory tradeoff is needed. The cap cannot compensate for a
GPU already occupied by another process; benchmark runs should use an
otherwise free device.

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

The five-way benchmark exposes the same three protocols through
`--five-way-image-mode`. Its prompt and message builder explicitly describe
only the images selected by that mode; the default remains
`marked_plus_crop` for backward compatibility.

An action-oriented ablation is available through
`--task-mode routing_four_way`. It preserves the original five-way labels in
the result JSONL while mapping them for scoring as follows:

- `aligned -> no_action`
- `wrong_object / unsupported -> relocate`
- `partial_coverage -> expand`
- `ambiguous -> tighten`

Select its image input with `--routing-image-mode`; the default is
`bbox_image_only`.

An option-likelihood variant avoids free-form generation and JSON parsing:
`--task-mode routing_option_likelihood`. The model sees one image, the object
reference, and the candidate box as absolute `xyxy` coordinates in the exact
Qwen smart-resized image frame. `--option-image-mode raw_image` uses the clean
scene; `bbox_image` uses the same scene with a red candidate rectangle.

The four routing actions are represented by the single-token completions
`A/B/C/D`. One multimodal forward pass produces all four next-token negative
log-likelihoods; the lowest-loss option is the prediction. Confidence is the
winning probability after normalizing only across those four options, and the
result metadata also records every option loss, token id, probability, and the
best-versus-second-best loss margin. The model execution contract lives in
`option_likelihood.py`, while image/coordinate preparation, prompts, and GQA
adaptation remain separate.

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
prompt construction, and model execution remain separate modules, so this
path does not alter the option-likelihood classifier.

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

For example:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  verifier.benchmarks.gqa_controlled.evaluator \
  --model-path weights/Qwen2.5-VL-7B-Instruct \
  --task-mode routing_option_likelihood \
  --option-image-mode raw_image \
  --split test \
  --output output/verifier_benchmark/qwen25_vl_7b/options_raw/results.jsonl \
  --verbose
```

Run the geometry-based variant with:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  verifier.benchmarks.gqa_controlled.evaluator \
  --model-path weights/Qwen2.5-VL-7B-Instruct \
  --task-mode routing_grounding_geometry \
  --grounding-image-mode raw_image \
  --grounding-accept-iou 0.5 \
  --grounding-containment 0.7 \
  --grounding-boundary-tolerance 1 \
  --grounding-prompt-protocol single_object_json_v2 \
  --split dev \
  --output output/verifier_benchmark/qwen25_vl_7b/geometry_raw/results.jsonl \
  --verbose
```

The same geometry task can use Grounding DINO as an independent reference
localizer. In this mode the detector sees only the clean original image and
object reference. Its highest-score detection is post-processed back to
absolute `xyxy` coordinates on the original image and compared directly with
the candidate; neither Qwen smart resize nor VoCoT square padding is involved.
No detection is recorded as an end-to-end localization failure rather than
being guessed as `relocate`.

Grounding DINO requires the dedicated `qwen25` environment (Transformers 4.49)
and a local checkpoint. Select box/text thresholds on `dev`, then freeze them
before evaluating `test`:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  verifier.benchmarks.gqa_controlled.evaluator \
  --model-path weights/grounding-dino-base \
  --task-mode routing_grounding_geometry \
  --geometry-backend grounding_dino \
  --grounding-image-mode raw_image \
  --grounding-accept-iou 0.5 \
  --grounding-containment 0.7 \
  --dino-box-threshold 0.30 \
  --dino-text-threshold 0.25 \
  --split dev \
  --output output/verifier_benchmark/gqa_controlled/routing_grounding_geometry/grounding_dino_base/dev/results.jsonl \
  --verbose
```

Run a small development smoke test in the dedicated Qwen environment:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/zhonggai/miniconda3/envs/qwen25/bin/python -u -m \
  verifier.benchmarks.gqa_controlled.evaluator \
  --split dev \
  --limit 20 \
  --output output/verifier_benchmark/qwen25_vl_3b/dev_smoke/results.jsonl \
  --verbose
```

After freezing the prompt, run the held-out test split by removing `--limit`
and changing `--split test`. The evaluator writes one resumable JSONL plus a
`.summary.json` containing five-way accuracy, macro-F1, per-class metrics,
confusion matrix, aligned-vs-invalid metrics, parse success, and self-reported
confidence summaries.

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
    verifier_log_path='output/verifier/events.jsonl',
    return_metadata=True,
)
```

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
