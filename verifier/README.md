# Pre-commit object--coordinate verifier

This folder implements the first, oracle-only verifier experiment. It does
not train, alter, or resize the generator, and it does not add tokenizer
tokens. The oracle file is the only verifier backend in this version.

## Oracle input

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

## Running it

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
