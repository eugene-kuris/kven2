# Kven II task integration workflow

## Deployment authority

For schema 2.0, `deployment-contract.json` at the exact feature commit is the
only deployment authority. The runner and integration command normalize it,
then hash compact key-sorted UTF-8 JSON (`separators=(",", ":")`, no trailing
newline). Result and integration manifests record the path and SHA-256. Inspect,
the immediate stage preflight, status, finalize, and rollback independently
re-read the exact committed blob and refuse any package-only or persisted-state
change.

## Trust boundary and lifecycle

`scripts/kven-integrate-task` consumes a Codex `result-manifest.json`, but uses the exact committed contract as authority. It is deliberately local and project-specific. It does not infer backup, migration, or restart scope from changed files.

The lifecycle is `inspect` → `dry-run` → `stage` → `AWAITING_ACCEPTANCE` → `finalize`. A stage failure or rejected acceptance instead leads to `rollback`. Stage never pushes. Only finalize, after explicit `--accept PASS`, pushes local main and verifies the local `origin/main` ref.

## Result manifest schema 2.0

The runner writes `result-manifest.json` with requested model (nullable), actual runtime model, actual reasoning effort, reported token usage (nullable), exit and final status, repository and branch identities, baseline and feature heads, commits, changed files, tests, diff-check and secret-scan results, network declaration, evidence provenance, embedded deployment contract, requested services/backups/migrations, and package path. Runtime facts are parsed from the Codex startup stream; a requested/actual model conflict forces `FAIL`. Prohibited network use is recorded as `false`; allowed but unobservable use is `null` with an explanation. A still-running bootstrap package uses `null` for unavailable final runtime rather than guessing and identifies `bootstrap_postprocessing`; normal future output identifies `automatic_runner`. `result-summary.md` is the canonical human entry point. Legacy text artifacts remain available, but schema 1.0 manifests are deliberately not integration-eligible under the stricter inspector.

`result_validation_tests` is the canonical runner contract. After Codex exits, the runner snapshots the exact feature branch, HEAD, index, and clean status, then executes every declared argv command itself. The state must remain identical after every command and the full set; a validation command may not commit, stage, switch branch, or create tracked/untracked data. The manifest always records the pre-validation feature HEAD. Each test record contains the exact name and redacted argv in contract order, start/finish timestamps, duration, exit code, pass/fail result, bounded output, and a result-package-relative full artifact path with size and SHA-256. Inspect resolves artifacts from the actual manifest directory, refuses absolute/traversal/outside/symlink paths, and verifies file size and checksum. Missing, extra, reordered, mismatched, malformed, timed-out, tampered, or failed evidence forces non-eligibility; tests are never inferred from Codex prose.

## Deployment contract schema 2.0

The repository-root `deployment-contract.json` requires `expected_baseline`, `feature_branch`, non-empty `allowed_paths`, `services`, `backups`, `migration_checks`, non-empty `result_validation_tests`, non-empty `pre_merge_tests`, non-empty `post_merge_tests`, `readiness_checks`, `fatal_log_checks`, `acceptance_checklist`, and `rollback`. A trailing slash in an allowed path denotes a subtree. Every backup item is an object with an absolute `path` and the declared `services` that must be stopped before restore.

A command item is an object containing a non-empty argv-array `command`, plus optional `name` and `timeout`. Shell strings and literal password/secret/token/API-key/access-token/Authorization/Bearer/private-key values are rejected without reflecting their values; commands must refer indirectly to protected environment or configuration sources. All three rollback guarantees must be `true`. A migration check declares `database`, application migration `command`, explicit `application_smoke_command`, and optional `cwd`, `database_env`, `idempotent`, `timeout`, and `verification_sql`. Both commands receive the disposable database path through the named environment variable. A temporary-table probe is not treated as application verification.

## Commands

```bash
/opt/kven2/scripts/kven-integrate-task inspect /path/to/codex-result --repository /opt/kven2
/opt/kven2/scripts/kven-integrate-task dry-run /path/to/codex-result --repository /opt/kven2
/opt/kven2/scripts/kven-integrate-task stage /path/to/codex-result --repository /opt/kven2
/opt/kven2/scripts/kven-integrate-task status /path/to/integration-run --repository /opt/kven2
/opt/kven2/scripts/kven-integrate-task finalize /path/to/integration-run --accept PASS --notes "accepted" --repository /opt/kven2
/opt/kven2/scripts/kven-integrate-task rollback /path/to/integration-run --notes "acceptance failed" --repository /opt/kven2
```

Inspect and dry-run are read-only. Production defaults to the trusted `/opt/kven2` boundary and rejects any manifest redirect. `--repository` is an explicit operator/test-fixture override, not a value inferred from the manifest. Inspect independently recomputes the exact baseline-to-feature commit set and changed paths, compares them to the manifest, reruns `git diff --check`, scans exact changed content offline, and verifies the feature worktree is clean at the recorded branch/head. Dry-run prints intended backups, exact merge commit, tests, restarts, readiness, and fatal-log checks.

## Stage and acceptance

Stage takes a per-repository lock and durable active-run marker, creates a run ID and evidence directory, persists state before every mutation, performs backups, validates migrations on disposable copies, runs tests, then reruns preflight immediately before mutation. It refuses branch/baseline/origin/worktree drift and merges the exact verified feature SHA with `--no-ff`; the mutable branch name is never the merge target. Merge conflicts abort without automatic resolution.

Immediately after the exact-SHA merge, stage records one clean branch/HEAD/index/status fingerprint and never replaces it with an observed state. That exact fingerprint is checked after every post-merge test, deployment command, service restart group, readiness command, fatal-log command, and immediately before acceptance. Any repository mutation is a failure. Only declared allowlisted services that were active before stage are restarted. Originally inactive services remain inactive. Exact textual systemd states are preserved; `failed`, transitional, and unknown states are refused before mutation. Activation uses bounded polling, failure detection, restart counts sampled before restart, after activation, after readiness/log checks, and during finalize. The expected exact service state and restart count are retained for acceptance and finalize verification.

Success stops at `AWAITING_ACCEPTANCE`. The summary gives the run ID, product-focused checklist, finalize command, and rollback command. Human acceptance evaluates behavior, not source patches. A future Telegram feature can ask the owner to send/reply/wait and judge continuity; automated evidence may include counts and identifiers, never message text.

## Backups and sensitive data

Sensitive backups default to `/var/backups/kven2/<run-id>/`, mode `0700`; database and file backups use `0600`. SQLite uses Python’s online backup API, which captures a consistent standalone database including committed WAL content, then records size/SHA-256 and runs `PRAGMA integrity_check`. Connections close in `finally` blocks. Restore first requires each backup-declared service to be inactive, revalidates backup size/checksum/integrity, atomically quarantines stale `-wal`/`-shm` sidecars inside the protected backup tree, replaces the main file, and verifies final integrity and checksum. Configuration restore revalidates its protected backup and preserves/verifies content checksum, mode, uid, and gid. Symlinks and non-regular sources, backups, targets, and sidecars are refused.

User-readable evidence defaults to `/home/eugene/kven-integration-results/<task>/<run-id>/`, directories `0755`, files `0644`. It contains metadata and redacted logs, never databases, `.env` contents, credentials, authentication files, or private message content. Obvious credential assignments are redacted. Terminal output is capped at 4,000 characters while full redacted command output remains in artifacts.

## Services, readiness, and failures

The allowlist is `agent-sandbox.service`, `kven2-main.service`, `kven2-telegram-gateway.service`, `kven-client-gateway.service`, and `email-agent.service`. Contracts list dependency-safe order; rollback stops in reverse order and starts in declared order. Readiness is explicit argv commands and must check more than active state where applicable. The Telegram gateway depends on main and agent-sandbox, so those dependencies precede it when all are restarted.

A read-only/pre-mutation automated failure records `AUTOMATED_CHECK_FAILED`, clears its active marker, and leaves Git and services untouched; protected partial backup or migration artifacts may remain as evidence. Once live mutation begins, failure attempts guarded rollback. Rollback compares branch, HEAD, index tree, tracked changes, and untracked files with the immutable expected state. Any difference produces `INTERNAL_ERROR` without `reset --hard`, preserving operator data. Only an exact recorded clean state may be reset; rollback then verifies baseline branch/HEAD/tree, clean status, and unchanged origin/main, restores protected backups, returns every service to its original active/inactive state, reruns checks, and records evidence. Stage exits are deterministic: 0 awaiting acceptance, 3 pre-mutation automated failure, 4 successful rollback after mutation, and 5 rollback refusal/internal error.

Finalize requires `AWAITING_ACCEPTANCE` and explicit `PASS`; verifies exact Git/service/restart-count state before checks, runs readiness and fatal-log checks under the same repository guard, then verifies exact Git/service/restart-count state again. Only after those pre-push guards does it record acceptance and push the exact staged SHA to `main`.

`git push` and every configured push hook are part of the executable final trust boundary. The run persists pending push evidence before execution, then records the exact redacted argv, timestamps, duration, exit code, bounded output, full redacted artifact, origin/main before and after, and whether the remote is known updated. If origin/main reaches the staged SHA, both a normal push and interrupted `FINALIZING` recovery enter the same post-push reconciliation: rebind the committed deployment contract, require the recorded clean branch/HEAD/index/tracked/untracked state and exact service/restart snapshot, rerun readiness and fatal-log checks under repository guards, recheck repository and services, and finally recheck origin/main. `FINALIZED` is persisted only after that proof succeeds.

If a failed push leaves origin/main at the baseline, the run remains retryable `FINALIZING`; the next attempt repeats all pre-push guards. An unexpected third origin/main value becomes non-retryable `INTERNAL_ERROR`. If origin/main is the staged SHA but reconciliation fails, the run records bounded recovery evidence as `FINALIZE_RECOVERY_REQUIRED`, clears the active marker, preserves local/operator data, refuses ordinary rollback and remote-history rewriting, and exits nonzero. Repeated `finalize` deliberately refuses this state; there is no automatic state-file override. The operator must preserve the evidence, restore the exact recorded local and service state through an independently reviewed recovery procedure, and obtain explicit recovery tooling or review rather than editing the manifest. Repeated finalize of an already `FINALIZED` run remains non-mutating and rejects later drift. Feature branch/worktree cleanup is intentionally manual because project convention treats deletion as destructive.

## Recovery and cleanup

Preserve integration and protected backup directories until the normal retention decision. The repository’s active-run marker prevents another stage while an integration awaits acceptance or a safe same-run retry. `FINALIZE_RECOVERY_REQUIRED` clears the marker because automatic continuation is forbidden. Use `status`, inspect the recorded phase and exact repository state, and use guarded `rollback` only before a remote update; never continue from an unexpected state. After finalize, inspect branch containment and remove a disposable worktree/branch only through the documented Codex workflow. If rollback cannot establish the guarded state, the run remains `INTERNAL_ERROR`; do not force-reset or overwrite live state. Use the evidence and normal Git/config/database recovery layers, with Veeam only as final operator-controlled recovery.
