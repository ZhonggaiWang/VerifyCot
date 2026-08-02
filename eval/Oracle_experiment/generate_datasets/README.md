# Verifier benchmark dataset generation

`build_gqa_verifier_benchmark.py` constructs a controlled object-coordinate
verification benchmark with five source construction subtypes. Active
evaluation is binary or four-way; `wrong_object` and `unsupported` both map to
the four-way `relocate` action.

The five construction rules live in separate modules under
`verifier_benchmark/`:

- `aligned.py`: slightly jittered target GT boxes;
- `wrong_object.py`: another differently named object's GT box;
- `partial_coverage.py`: a 25--50% crop inside the target;
- `ambiguous.py`: one union box enclosing the target and a different object;
- `unsupported.py`: a random region avoiding retained annotated foreground
  objects.

Every rendered image uses the same magenta outline without label-dependent
text, color, or fill. The model-facing view is stored in `model_input` and
contains only the canonical object reference and rendered image path. GT,
construction details, and geometry remain in the outer record for offline
evaluation and auditing.

Benchmark datasets are stable, versioned inputs rather than experiment runs.
New versions belong under:

```text
output/verifier_benchmark/datasets/gqa_controlled/<version>/
```

Evaluation results must instead use
`output/verifier_benchmark/runs/<split>/<study>/<method>/<setting>/<run_id>/`.
The existing controlled V1 tree at
`output/verifier_benchmark/gqa_controlled/v1/` is a frozen historical input and
is not moved, so recorded commands and consumers remain reproducible.

Example smoke build:

```bash
python eval/Oracle_experiment/generate_datasets/build_gqa_verifier_benchmark.py \
  --output-dir output/verifier_benchmark/datasets/gqa_controlled/smoke_20260801 \
  --count-per-class 5
```

Example build for a new full version:

```bash
python eval/Oracle_experiment/generate_datasets/build_gqa_verifier_benchmark.py \
  --output-dir output/verifier_benchmark/datasets/gqa_controlled/v2 \
  --count-per-class 300 \
  --seed 20260729 \
  --dev-fraction 0.2
```

The builder refuses to overwrite an existing benchmark directory. Use a new
output directory for every version. `unsupported` records retain an explicit
annotation-completeness caveat and should be manually spot-audited before
reporting final metrics.
