# Kven II task integration workflow

## Trust boundary and lifecycle

`scripts/kven-integrate-task` consumes a Codex `result-manifest.json` and an embedded deployment contract. It is deliberately local and project-specific. It does not infer backup, migration, or restart scope from changed files.

The lifecycle is `inspect` → `dry-run` → `stage` → `AWAITING_ACCEPTANCE` → `finalize`. A stage failure or rejected acceptance instead leads to `rollback`. Stage never pushes. Only finalize, after explicit `--accept PASS`, pushes local main and verifies the local `origin/main` ref.

## Result manifest schema 1.0

The runner writes `result-manifest.json` with task/timing/model data, exit and final status, repository and branch identities, baseline and feature heads, commits, changed files, tests, diff-check and secret-scan results, network declaration, embedded deployment contract, requested services/backups/migrations, and package path. Unknown token usage and network use are JSON `null`, never guessed from prose. `result-summary.md` is the canonical human entry point. Legacy text artifacts remain compatible.

## Deployment contract schema 1.0

The repository-root `deployment-contract.json` requires `expected_baseline`, `feature_branch`, non-empty `allowed_paths`, `services`, `backups`, `migration_checks`, `pre_merge_tests`, `post_merge_tests`, `readiness_checks`, `acceptance_checklist`, and `rollback`. A trailing slash in an allowed path denotes a subtree.

A command item is an argv JSON array or an object with `command` and optional `timeout`. A migration check declares `database`, `command`, optional `cwd`, `database_env`, `idempotent`, `timeout`, and `verification_sql`. The command receives the disposable database path through the named environment variable.

## Commands

```bash
scripts/kven-integrate-task inspect /path/to/codex-result
scripts/kven-integrate-task dry-run /path/to/codex-result
scripts/kven-integrate-task stage /path/to/codex-result
scripts/kven-integrate-task status /path/to/integration-run
scripts/kven-integrate-task finalize /path/to/integration-run --accept PASS --notes "accepted"
scripts/kven-integrate-task rollback /path/to/integration-run --notes "acceptance failed"
```

Inspect and dry-run are read-only. Inspect verifies successful Codex status, recorded tests, diff/secret checks, repository identity, clean and exact main/origin-main baseline, exact feature head and commit set, ancestry, allowed paths, and contract completeness. Dry-run prints intended backups, merge, tests, restarts, and readiness checks.

## Stage and acceptance

Stage creates a durable run ID and evidence directory, performs backups, validates migrations on disposable copies, runs tests, merges with `--no-ff` into local main, executes declared repository-local deployment steps, restarts declared allowlisted services, and runs readiness commands. Merge conflicts abort without automatic resolution.

Success stops at `AWAITING_ACCEPTANCE`. The summary gives the run ID, product-focused checklist, finalize command, and rollback command. Human acceptance evaluates behavior, not source patches. A future Telegram feature can ask the owner to send/reply/wait and judge continuity; automated evidence may include counts and identifiers, never message text.

## Backups and sensitive data

Sensitive backups default to `/var/backups/kven2/<run-id>/`, mode `0700`; database and file backups use `0600`. SQLite uses Python’s online backup API, which captures a consistent database including committed WAL content, then runs `PRAGMA integrity_check`. Connections close in `finally` blocks. Configuration uses `copy2`, records mode/owner/checksum, and restores through a same-directory temporary file plus atomic replacement.

User-readable evidence defaults to `/home/eugene/kven-integration-results/<task>/<run-id>/`, directories `0755`, files `0644`. It contains metadata and redacted logs, never databases, `.env` contents, credentials, authentication files, or private message content. Obvious credential assignments are redacted. Terminal output is capped at 4,000 characters while full redacted command output remains in artifacts.

## Services, readiness, and failures

The allowlist is `agent-sandbox.service`, `kven2-main.service`, `kven2-telegram-gateway.service`, `kven-client-gateway.service`, and `email-agent.service`. Contracts list dependency-safe order; rollback stops in reverse order and starts in declared order. Readiness is explicit argv commands and must check more than active state where applicable. The Telegram gateway depends on main and agent-sandbox, so those dependencies precede it when all are restarted.

Any automated failure attempts guarded rollback. Rollback refuses an unexpected Git head, stops declared services, resets only to the recorded pre-stage head, restores declared backups atomically, restarts services, and records evidence. An interrupted run can be inspected with `status`; if its Git head remains the recorded baseline or staged head, use `rollback`. Otherwise investigate without resetting.

Finalize verifies the run remains `AWAITING_ACCEPTANCE`, main is clean and still at the staged head, records acceptance, pushes main, and verifies origin/main. Repeated finalize returns the finalized state. Feature branch/worktree cleanup is intentionally manual because project convention treats deletion as destructive.

## Recovery and cleanup

Preserve integration and protected backup directories until the normal retention decision. After finalize, inspect branch containment and remove a disposable worktree/branch only through the documented Codex workflow. If rollback cannot establish the guarded state, the run becomes `INTERNAL_ERROR`; do not force-reset or overwrite live state. Use the evidence and normal Git/config/database recovery layers, with Veeam only as final operator-controlled recovery.
