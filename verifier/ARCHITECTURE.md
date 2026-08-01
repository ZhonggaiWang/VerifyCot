# Verifier and expert routing architecture

The controller recognizes three public system roles:

- `ActionVerifierBackend` judges an uncommitted object–coordinate claim and
  returns the canonical four-action output.
- `GrounderBackend` relocates an object after `wrong_object`.
- `BoxRefinerBackend` expands or tightens an existing region.

`BoxPredictor` is an internal reusable model capability, not a fourth public
role called “localizer.” The same Qwen or Grounding DINO prediction capability
can support a geometry verifier or a Grounder adapter without coupling those
roles in the controller.

All active verifiers normalize to the versioned `vocot_four_action_v1`
contract:

```text
P(no_action), P(relocate), P(expand), P(tighten)
```

`predicted_action` is the highest-probability action. `unknown`,
`unsupported`, runtime failure, and confidence rejection are represented as
`abstained=true`, not as additional visual-error classes. The routing policy
then chooses fail-open `no_action` or system-level `abstain`. Historical
hard-label verifiers set `action_probabilities=null` and record
`probability_source=unavailable_*`; they never manufacture fake softmax
probabilities.

The pre-commit path is:

```text
VoCoT candidate </coor>, before REFbind
  -> ActionVerifierBackend
  -> RoutingPolicy
       no_action               -> accept candidate
       relocate                -> GrounderBackend
       expand                  -> BoxRefinerBackend
       tighten                 -> BoxRefinerBackend
       abstained/low confidence-> configurable fail-open or stop
  -> ExpertRouter
  -> clean replay of only the selected coordinate
  -> normal REFbind
```

Relevant packages:

- `contracts/`: stable public role interfaces.
- `routing_policy.py`: pure decision-to-action policy.
- `expert_router.py`: specialist dispatch only.
- `models/`: reusable Qwen/DINO model capabilities.
- `verifier_backends/`: role-specific verifier implementations and
  compositions; reusable model execution remains under `models/`.
- `experts/grounders/` and `experts/refiners/`: correction-role adapters.
- `oracle_targets.py`: shared conservative reference-to-GT resolution used
  only by oracle upper-bound components.
- `runtime/`: model-agnostic JSONL transport.
- `workers/endpoints/`: JSON adapters for individual roles.
- `workers/qwen_dino_worker.py`: deployable composition only.
- `workers/dino_geometry_worker.py`: role-specific remote DINO verifier.

For split-GPU inference, `PersistentJsonlWorkerClient` owns the verifier
subprocess and `RemoteActionVerifierBackend` exposes it as the same
`ActionVerifierBackend` consumed by the controller. The worker receives an
image path, object reference, and padded-normalized candidate; DINO sees only
the clean original image. Its predicted original-pixel box is mapped back into
VoCoT's padded coordinate frame before geometry routing.

Oracle experts do not require verifier access to GT annotations.
`oracle_targets.OracleTargetResolver` independently resolves the latest
unique explicit alias from `sample_context["oracle_targets"]`. This permits a
learned or DINO verifier to route to
`experts.grounders.OracleGrounderBackend` and
`experts.refiners.OracleBoxRefinerBackend` without contaminating the verifier
output with GT. The IoU oracle verifier separately lives in
`verifier_backends/oracle.py`.

The `refine` worker operation is reserved but deliberately returns
`box_refiner_not_configured` until a real expand/tighten expert is selected.
