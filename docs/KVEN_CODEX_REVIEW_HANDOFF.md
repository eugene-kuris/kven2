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
scripts/kven-codex-task TASK.txt \
  --review-findings /safe/path/review-findings.json \
  --previous-result /home/eugene/kven-codex-results/PREVIOUS_RUN \
  --previous-feature-sha REVIEWED_FEATURE_SHA
```

The runner rejects malformed, duplicate-ID, symlink, non-file, and protected
`/agent/data` inputs. It copies the validated file into the result package and
does not mutate the source. The prompt names every finding and preserve
constraint. Ordinary Git, test, contract, handoff, and secret safeguards still
apply. The runner verifies the previous package, handoff, branch ref, clean
worktree, reviewed SHA, task ID/hash, findings binding, current-main ancestry,
and prior test/requirement evidence before Codex starts. `correction-context.json`
contains the original task identity, previous handoff, commits, test evidence,
requirement map, decisions, findings, and review state. If the prior manifest has
a session ID, Codex CLI 0.146.1 supports `codex exec resume SESSION_ID -` and the
runner uses it; `--no-resume` deterministically exercises the complete-context
fallback. `correction_results` must map every supplied finding exactly once to root
cause, correction, changed files/symbols, tests, verification, remaining risk,
and `FIXED`, `PARTIAL`, `NOT_FIXED`, or `REJECTED_WITH_REASON` status.

The canonical source is `result-manifest.json`. A bootstrap correction package
created while this capability was introduced may omit that manifest only when it
has a schema-valid `reviewer-context.json` and matching complete JSON/Markdown
handoff. The runner validates the package path, task digest, branch, worktree,
feature SHA, commits, changed paths, tests, decisions, requirements, review state,
and live Git before execution. Missing or conflicting facts fail closed; none are
guessed. The new `correction-context.json` records `previous_result_source` as
either `result-manifest` or `bootstrap-reviewer-context`.

File-backed task SHA-256 values cover the exact bytes on disk, including CRLF.
For stdin, the canonical bytes are the UTF-8 encoding of the received text.

Every correction also produces `delta-handoff.json` and Markdown. The delta
binds the previous run/SHA and new SHA, changed paths, tests added/run, preserved
decisions, invalidated assumptions, deployment/migration/restart deltas, new
risks, and an exact partition of open and closed finding IDs. Prior packages and
handoffs remain immutable.

## Reviewer context, bundle, and status

After all runner gates, each new package contains `reviewer-context.json`,
`review-status.json`, and mode-0644 `chatgpt-review-bundle.md`. Context holds
exact task hash, run/session identity, Git state, paths, services, tests, scans,
network use, lineage, sequence, risk, and review status. The standalone bundle
contains the final manifest status, handoff, requirement/test map, correction
history, risks, and recommended checks without embedding large logs.

```bash
scripts/kven-review-handoff RESULT status --json
```

Status reports run number, before/after SHA, runtime/tokens, commits/tests,
received/closed/open findings, latest handoff/bundle, resumability, and final
evidence-scan result.

If either evidence scan changes the final outcome, the runner rewrites the
manifest, result summary, reviewer context, review status, standalone bundle,
exit code, and plain summary to FAIL before returning. A terminal scan therefore
cannot leave a stale PASS claim on another status surface.

## Roles and workflow

- The architectural session defines architecture and task boundaries.
- The working session reads the canonical bundle first, validates exact Git and
  high-risk evidence, then integrates or creates structured findings.
- The Codex developer implements and corrects findings on the same lineage.
- An optional fresh Codex reviewer may independently inspect the bundle and exact
  code; it does not share the developer's assumptions.
- A future control session may route project work; it is not implemented here.

Manual patching is appropriate only for a trivial, obvious correction where the
correction workflow would cost more than independent verification. Otherwise:
initial run → handoff/bundle → review findings → correction run → delta review.
The reviewer starts with the bundle, not broad repository rediscovery.

Optional sanitized read-only HTTPS publication remains out of scope. If later
approved, export only explicit handoff/context/bundle artifacts behind opaque
run IDs, with no directory listing, source tree, credentials, environment files,
databases, private messages, or arbitrary path access.

## Compatibility boundary

Packages whose manifest provenance is `automatic_runner` are new packages and
must have a passing schema-1.0 `reviewer_handoff` validation record to be
integration-eligible. Historical schema-2.0 packages with
`bootstrap_postprocessing` provenance remain readable and inspectable without
retroactive handoff enforcement. The integration inspector surfaces either the
validated record or the historical compatibility state; it does not perform
semantic review.
