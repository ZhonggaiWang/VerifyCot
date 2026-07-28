# Legacy one-shot repair runs

One-shot runs created before `full_238_padding_fix` used a reference manifest
derived from incorrectly normalized VStar GT boxes. Existing dated prompt and
sandbox ablation outputs are retained for development history, but their GT
recovery metrics are deprecated.

New experiments must use:

`output/vstar/one_shot_reference_repair/full_238_padding_fix/manifest.jsonl`
