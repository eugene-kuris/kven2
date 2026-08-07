# Codex reviewer handoff and correction loop

Handoff schema `1.0` gives a reviewing working session the state established by
Codex without treating that state as a substitute for independent review. For
results created by the OPS-003A-capable `kven-codex-task`,
`handoff-to-reviewer.json` is mandatory and authoritative;
`handoff-to-reviewer.md` is its human-readable rendering.

## Handoff content and evidence

The JSON contains `schema_version`, `task_identity`, `task_understood_as`,
`implementation_map`, `important_design_decisions`, `rejected_alternatives`,
`architecture_deviations`, `data_schema_migration_changes`,
`runtime_behavior_changed`, `existing_behavior_intentionally_preserved`,
`known_weak_points`, `uncertainties`, `requirement_evidence_map`, `tests`,
`exact_git_state`, `recommended_reviewer_checks`,
`do_not_spend_time_rediscovering`, `unresolved_issues`, and
`correction_routing_metadata`. Non-applicable list sections remain present as
explicit empty lists. A corrective handoff also contains `correction_results`.

Claims should cite a path and symbol, exact test name, result artifact, manifest
field, or commit. The runner validates identity, baseline, branch, exact feature
HEAD, ordered commits, changed files, implementation-map coverage, required
sections, clean Git state, and Markdown task/HEAD agreement. Both handoff files
are covered by the ordinary result-package evidence secret scan.

The reviewer may trust the runner's manifest record as evidence that these
syntactic and exact-fact checks passed. The reviewer should independently judge
the task interpretation, design choices, adequacy of tests, correctness of
claims, and residual risk. Start with `recommended_reviewer_checks`.

## Reading a handoff

The helper is read-only and accepts a package directory or the JSON path:

```bash
scripts/kven-review-handoff RESULT summary
scripts/kven-review-handoff RESULT recommended-review-checks
scripts/kven-review-handoff RESULT requirement-evidence --json
scripts/kven-review-handoff RESULT uncertainties-weak-points
scripts/kven-review-handoff RESULT correction-results
```

## Reviewer findings and correction

`review-findings.json` schema `1.0` has `schema_version`, `task_id`, and a
`findings` list. Each finding has a unique `REV-NNN` `finding_id`, `severity`
(`blocking`, `major`, `minor`, or `note`), `status`, `claim_or_requirement`,
`observed`, `evidence`, `required_correction`, `must_preserve`, and
`verification_required`; `reviewer_notes` is optional.

Invoke a corrective run with:

```bash
scripts/kven-codex-task TASK.txt --review-findings /safe/path/review-findings.json
```

The runner rejects malformed, duplicate-ID, symlink, non-file, and protected
`/agent/data` inputs. It copies the validated file into the result package and
does not mutate the source. The prompt names every finding and preserve
constraint. Ordinary Git, test, contract, handoff, and secret safeguards still
apply. `correction_results` must map every supplied finding exactly once to root
cause, correction, changed files/symbols, tests, verification, remaining risk,
and `FIXED`, `PARTIAL`, `NOT_FIXED`, or `REJECTED_WITH_REASON` status.

## Compatibility boundary

Packages whose manifest provenance is `automatic_runner` are new packages and
must have a passing schema-1.0 `reviewer_handoff` validation record to be
integration-eligible. Historical schema-2.0 packages with
`bootstrap_postprocessing` provenance remain readable and inspectable without
retroactive handoff enforcement. The integration inspector surfaces either the
validated record or the historical compatibility state; it does not perform
semantic review.
