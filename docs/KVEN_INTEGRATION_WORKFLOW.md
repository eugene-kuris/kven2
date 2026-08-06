# Kven II task integration workflow

## Trust boundary and lifecycle

`scripts/kven-integrate-task` consumes a Codex `result-manifest.json` and an embedded deployment contract. It is deliberately local and project-specific. It does not infer backup, migration, or restart scope from changed files.

The lifecycle is `inspect` → `dry-run` → `stage` → `AWAITING_ACCEPTANCE` → `finalize`. A stage failure or rejected acceptance instead leads to `rollback`. Stage never pushes. Only finalize, after explicit `--accept PASS`, pushes local main and verifies the local `origin/main` ref.

## Result manifest schema 2.0

The runner writes `result-manifest.json` with task/timing/model data, exit and final status, repository and branch identities, baseline and feature heads, commits, changed files, tests, diff-check and secret-scan results, network declaration, embedded deployment contract, requested services/backups/migrations, and package path. Unknown token usage and network use are JSON `null`, never guessed from prose. `result-summary.md` is the canonical human entry point. Legacy text artifacts remain available, but schema 1.0 manifests are deliberately not integration-eligible under the stricter inspector.

`result_validation_tests` is the canonical runner contract. After Codex exits, the runner executes every declared argv command itself. Each test record contains its redacted argv, start/finish timestamps, duration, exit code, pass/fail result, bounded output, and a full redacted artifact path. Missing, empty, malformed, timed-out, or failed validation tests force `final_codex_status` to `FAIL`; tests are never inferred from Codex prose.

## Deployment contract schema 2.0

The repository-root `deployment-contract.json` requires `expected_baseline`, `feature_branch`, non-empty `allowed_paths`, `services`, `backups`, `migration_checks`, non-empty `result_validation_tests`, non-empty `pre_merge_tests`, non-empty `post_merge_tests`, `readiness_checks`, `fatal_log_checks`, `acceptance_checklist`, and `rollback`. A trailing slash in an allowed path denotes a subtree. Every backup item is an object with an absolute `path` and the declared `services` that must be stopped before restore.

A command item is an object containing a non-empty argv-array `command`, plus optional `name` and `timeout`. Shell strings are rejected. A migration check declares `database`, application migration `command`, explicit `application_smoke_command`, and optional `cwd`, `database_env`, `idempotent`, `timeout`, and `verification_sql`. Both commands receive the disposable database path through the named environment variable. A temporary-table probe is not treated as application verification.

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

Only declared allowlisted services that were active before stage are restarted. Originally inactive services remain inactive. Activation uses bounded polling, failure detection, restart-count loop detection, readiness commands, and explicit bounded fatal-log commands. The exact active/inactive state is retained for rollback and finalize verification.

Success stops at `AWAITING_ACCEPTANCE`. The summary gives the run ID, product-focused checklist, finalize command, and rollback command. Human acceptance evaluates behavior, not source patches. A future Telegram feature can ask the owner to send/reply/wait and judge continuity; automated evidence may include counts and identifiers, never message text.

## Backups and sensitive data

Sensitive backups default to `/var/backups/kven2/<run-id>/`, mode `0700`; database and file backups use `0600`. SQLite uses Python’s online backup API, which captures a consistent standalone database including committed WAL content, then records size/SHA-256 and runs `PRAGMA integrity_check`. Connections close in `finally` blocks. Restore first requires each backup-declared service to be inactive, revalidates backup size/checksum/integrity, atomically quarantines stale `-wal`/`-shm` sidecars inside the protected backup tree, replaces the main file, and verifies final integrity and checksum. Configuration restore revalidates its protected backup and preserves/verifies content checksum, mode, uid, and gid. Symlinks and non-regular sources, backups, targets, and sidecars are refused.

User-readable evidence defaults to `/home/eugene/kven-integration-results/<task>/<run-id>/`, directories `0755`, files `0644`. It contains metadata and redacted logs, never databases, `.env` contents, credentials, authentication files, or private message content. Obvious credential assignments are redacted. Terminal output is capped at 4,000 characters while full redacted command output remains in artifacts.

## Services, readiness, and failures

The allowlist is `agent-sandbox.service`, `kven2-main.service`, `kven2-telegram-gateway.service`, `kven-client-gateway.service`, and `email-agent.service`. Contracts list dependency-safe order; rollback stops in reverse order and starts in declared order. Readiness is explicit argv commands and must check more than active state where applicable. The Telegram gateway depends on main and agent-sandbox, so those dependencies precede it when all are restarted.

Any automated failure attempts guarded rollback. Rollback compares branch, HEAD, index tree, tracked changes, and untracked files with the exact state persisted after the last Git mutation. Any difference produces `INTERNAL_ERROR` without `reset --hard`, preserving operator data. Only an exact recorded clean state may be reset; rollback then verifies baseline branch/HEAD/tree, clean status, and unchanged origin/main, restores protected backups, returns every service to its original active/inactive state, reruns checks, and records evidence.

Finalize requires `AWAITING_ACCEPTANCE` and explicit `PASS`; verifies the exact recorded Git state, baseline origin relationship, services, readiness, and fatal logs; records acceptance; then pushes the exact staged SHA to `main` and verifies origin/main. A persisted `FINALIZING` state makes interrupted push recovery deterministic. Repeated finalize verifies and returns the finalized state. Feature branch/worktree cleanup is intentionally manual because project convention treats deletion as destructive.

## Recovery and cleanup

Preserve integration and protected backup directories until the normal retention decision. The repository’s active-run marker prevents another stage while an integration awaits acceptance or recovery. Use `status`, inspect the recorded phase and exact repository state, and use guarded `rollback`; never continue from an unexpected state. After finalize, inspect branch containment and remove a disposable worktree/branch only through the documented Codex workflow. If rollback cannot establish the guarded state, the run remains `INTERNAL_ERROR`; do not force-reset or overwrite live state. Use the evidence and normal Git/config/database recovery layers, with Veeam only as final operator-controlled recovery.
