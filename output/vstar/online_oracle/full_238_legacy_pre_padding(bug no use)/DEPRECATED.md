# Deprecated: pre-padding-coordinate result

This run is retained only for historical reproducibility.

Its VStar GT boxes were normalized on the original rectangular image but were
passed directly to a model that consumes a center-padded square image. The
forced coordinate text and REFbind binding were internally consistent, but
they referred to the wrong model-image coordinate system.

Do not use this run for oracle accuracy, grounding IoU, repair manifests, or
new experiments. Use:

`output/vstar/online_oracle/full_238_padding_fix/results.jsonl`
