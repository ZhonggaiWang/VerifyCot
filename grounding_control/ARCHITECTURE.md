# Grounding-control architecture

`grounding_control` owns the full object--coordinate control loop around
VoCoT generation.  Verification is one role in that loop; the package also
owns routing policy, expert dispatch, clean coordinate replay, worker
transport, and experiment adapters.

## Primary binary path

The primary verifier contract is `AlignmentVerifierBackend`.  It receives the
clean image, current object reference, and one uncommitted candidate box, then
returns an `AlignmentVerifierOutput` whose score always has the same
direction: larger means better alignment.  Every score declares its kind and
backend-specific semantics; raw IoU proxies and self-reported confidence are
not presented as calibrated probabilities.

```text
VoCoT emits <coor>q</coor>, before q enters REFbind
  -> AlignmentVerifierBackend
  -> AlignmentRoutingPolicy
       score >= tau_accept                 -> accept q
       score <= tau_reject                 -> call Grounder
       tau_reject < score < tau_accept     -> uncertain
       verifier failure                    -> fail-open accept q
  -> first-version uncertain policy calls Grounder
  -> commit exactly q or the Grounder box
  -> clean replay through normal REFbind
  -> freely continue the CoT
```

The `reject` and `uncertain` bands remain distinct in logs even though the
first version routes both to the Grounder.  This preserves the interface for a
future strong checker or abstention policy without changing the verifier
contract.

## Public roles

- `AlignmentVerifierBackend` scores whether a candidate supports its object
  reference.  It never performs correction.
- `GrounderBackend` independently relocates the referenced object.
- `PrecommitGroundingController` implements the pre-commit generation and
  clean-commit loop.

Reusable Qwen and Grounding DINO execution lives under `models/`.  A model
capability may support both verifier and Grounder adapters, but the public
roles remain separate and the verifier never silently invokes an expert.

## Retained four-way path

The canonical four-action output
`no_action / relocate / expand / tighten` is retained under the explicit
`four_way/` namespace for appendix diagnostics and reproduction. It is not
imported by the primary binary package surface. No action semantics or
archived wire schemas changed during the move.

Historical prompt-based self-repair, retry loops, stored-oracle lookup, and
sandbox REFbind ablations remain under `legacy/`.

## Current package responsibilities

- `contracts/`: stable binary verifier and Grounder interfaces.
- `core/coordinate_rollout.py`: policy-neutral generation boundary and clean
  coordinate replay.
- `core/precommit_controller.py`: binary verification, routing, Grounder
  selection, and clean coordinate commitment.
- `core/alignment_policy.py`: binary dual-threshold policy.
- `core/calibration.py`: optional alignment-score calibration.
- `core/expert_dispatch.py`: expert invocation after policy selection.
- `verifiers/`: binary verifier implementations. Its Qwen path uses
  `backend.py`, `classifier.py`, `prompt.py`, `parser.py`, and `inputs.py`.
- `experts/grounders/`: binary-mainline correction adapters.
- `four_way/`: archived action contracts, controller, policy, verifiers, and
  action workers. Every non-accept action now routes to a Grounder.
- `models/`: reusable Qwen and Grounding DINO model capabilities.
- `transport/`: model-agnostic JSONL transport.
- `workers/`: persistent binary verifier and role-specific Grounder processes,
  including standalone `dino_grounder` and `qwen_grounder` entry points;
  archived action workers live under `four_way/workers/`.
- `benchmarks/`: controlled verifier evaluation.
- `legacy/`: archived prompt-repair experiments.

## Oracle isolation

Oracle experts resolve targets independently through `OracleTargetResolver`.
A learned or DINO verifier can therefore route to an oracle Grounder without
receiving GT annotations itself.  Oracle verifier outputs and oracle expert
outputs remain separate experimental components.

## Removed compatibility surface

Phase three removed the deprecated top-level `verifier` facade and duplicate
root modules such as `types.py`, `prompts.py`, `routing_policy.py`,
`verifier_backends/`, and `runtime/`. Canonical imports are now explicit:
binary code uses `grounding_control`, four-action experiments use
`grounding_control.four_way`, historical prompt repair uses
`grounding_control.legacy`, and JSONL transport uses
`grounding_control.transport`.
