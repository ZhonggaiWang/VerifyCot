# Legacy output index

Everything listed here predates the canonical `runs/` hierarchy and remains at
its current path so scripts, manifests, and recorded comparisons do not break.
Do not add new results to these trees unless reproducing an archived command.

This index maps historical locations to the taxonomy used by new runs; it is
not a claim that every smoke or interrupted run is publication-ready.

The upstream `eval/evaluate_benchmark.py` + `eval/merge_benchmark.py` +
`eval/commands/run_metric.sh` pipeline is also intentionally preserved as a
legacy protocol. Its rank-local JSON names and downstream metric paths are
tightly coupled. New grounding-control runs use the canonical hierarchy;
reproducing the original VoCoT benchmark commands should continue to pass an
explicit `--output_dir` and retain their archived layout.

## Stable inputs (not runs)

| Historical path | Role | Policy |
| --- | --- | --- |
| `vstar/annotations/` | VStar manifests and oracle boxes | Keep as stable input |
| `gqa/annotations/` | GQA manifests and oracle subsets | Keep as stable input |
| `verifier_benchmark/gqa_controlled/v1/` | Controlled verifier benchmark data/results | Preserve; versioned consumers may depend on it |

## VStar historical runs

| Historical path | Canonical study/method interpretation |
| --- | --- |
| `vstar_baseline_independent_20260724/` | `full_238/baseline/volcano_7b/default` |
| `vstar_counterfactual_sweeps/` | `full_238/counterfactual/<intervention>/<position>` |
| `vstar/counterfactual/` | `full_238/counterfactual/<intervention>/<position>` |
| `vstar_online_oracle/` | `full_238/oracle/always_gt/default` |
| `vstar/online_oracle/` | `full_238/oracle/always_gt/<setting>` |
| `vstar/selective_oracle_router/` | `full_238/routing/oracle_verifier__oracle_experts/<iou>` |
| `vstar/dino_geometry_oracle_experts/` | `full_238/routing/dino_geometry__oracle_experts/<iou>` |
| `vstar/one_shot_reference_corruption/` | `full_238_matchable_198/counterfactual/one_shot_reference_corruption/random_box` |
| `vstar/one_shot_reference_repair/` | `full_238_matchable_198/repair/one_shot/<setting>` |
| `vstar/natural_error_repair/` | `full_238/repair/natural_error/<setting>` |
| `vstar/compatibility/` | `full_238/compatibility/<environment>/default` |

The following locations are support artifacts rather than experiment runs:

- `vstar/baseline_grounding_renders/`
- `vstar/repair_renders/`
- `vstar_oracle_box_audit/`

Known invalid/deprecated coordinate-system results remain visible for audit but
must not be used in comparisons:

- `vstar/online_oracle/full_238_legacy_pre_padding(bug no use)/`
- `vstar/one_shot_reference_repair/full_238_legacy_pre_padding/`
- any run containing its own `DEPRECATED.md`, or listed by
  `vstar/one_shot_reference_repair/DEPRECATED_PRE_PADDING_RUNS.md`

The padding-fixed references currently used by later experiments include:

- `vstar/online_oracle/full_238_padding_fix/`
- `vstar/one_shot_reference_repair/full_238_padding_fix/`

## GQA historical runs

| Historical path | Canonical study/method interpretation |
| --- | --- |
| `gqa/testdev_baseline/` | `testdev_12578/baseline/volcano_7b/default` |
| `gqa/counterfactual/` | split-specific `counterfactual/<intervention>/<position>` |
| `gqa/full_intervention_suite/` | a legacy suite containing multiple counterfactual methods under one tag |
| `gqa/online_oracle/` | split-specific `oracle/always_gt/<setting>` |
| `gqa/selective_oracle_router/` | split-specific `routing/oracle_verifier__oracle_experts/<iou>` |

GQA's nested `shards/shard_NNN/` directories are data-parallel fragments of
the surrounding logical run.  The merged files in the surrounding run root
are authoritative.  They should not be migrated into separate canonical run
IDs.

## Verifier benchmark historical runs

| Historical path | Canonical interpretation |
| --- | --- |
| `verifier_benchmark/qwen25_vl_3b/` | controlled benchmark runs for the Qwen 3B backend |
| `verifier_benchmark/qwen25_vl_7b/` | controlled benchmark runs for the Qwen 7B backend |
| `verifier_benchmark/routing_four_way_action_prompt_native_max/` | four-way routing protocol comparison |
| `verifier_benchmark/backend_cleanup_regression/` | backend refactor regression evidence |
| `verifier_benchmark/gqa_controlled/` except stable `v1/` | controlled benchmark smoke/protocol results |

New benchmark results should follow the same run contract, for example:

```text
output/verifier_benchmark/runs/gqa_controlled_v1_<dev|test|all>/
  binary/<backend>/<protocol>/<run_id>/
  four_way/<backend>/<protocol>/<run_id>/
  threshold_search/<backend>/<setting>/<run_id>/
```

The exact six-level generic form remains
`<dataset>/runs/<split>/<study>/<method>/<setting>/<run_id>`; the tree above
uses benchmark task/protocol names for those dimensions.
