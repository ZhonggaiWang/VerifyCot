# Experiment output layout

New VStar runs should use the following hierarchy.  Existing historical output
folders are intentionally left untouched.

```text
output/
  gqa/
    annotations/
      oracle_val_1000/
        manifest.jsonl
        manifest.summary.json
    counterfactual/
      <intervention_mode>/<position>/<run_tag>/
        shards/shard_000/
        results.jsonl
        results.summary.json
    online_oracle/
      <run_tag>/
    full_intervention_suite/
      <run_tag>/
        remove_grounding/{random,first,last}_position/
        random_box/{random,first,last}_position/
        online_oracle/
  vstar/
    baseline/
      <run_tag>/
    counterfactual/
      random_box/
        <run_tag>/
          random_position/run_01/
          first_position/run_01/
          last_position/run_01/
      remove_grounding/
        <run_tag>/
          random_position/run_01/
          first_position/run_01/
          last_position/run_01/
    online_oracle/
      <run_tag>/
    annotations/
      oracle_boxes/
```

Each run directory contains `results.jsonl`, `results.summary.json`, and, for
scheduler-based sweeps, one `run.log` per shard. `run_tag` is normally a
timestamp and may be overridden with `RUN_TAG=...` for reproducibility.

The counterfactual scheduler defaults to `random_box`.  Set
`INTERVENTION_MODE=remove_grounding` to run the same random/first/last position
strategies while suppressing the selected `<coor>` and its REFbind feature.

GQA's `scripts/run_gqa_counterfactual.sh` and
`scripts/run_gqa_online_oracle.sh` split the fixed manifest across idle cards
and merge all shard JSONLs back into the run root.
