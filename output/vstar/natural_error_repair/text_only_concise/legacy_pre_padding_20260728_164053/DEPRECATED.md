# Deprecated: natural-error audit used pre-padding GT coordinates

This run selected natural grounding errors against VStar GT boxes in the
original-image coordinate system, while VoCoT operates on a center-padded
square image. Some checker triggers and all reported GT IoUs are therefore
not valid as the final experiment.

It is retained only for historical analysis. Rerun against:

`output/vstar/online_oracle/full_238_padding_fix/results.jsonl`
