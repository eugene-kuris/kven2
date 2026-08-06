# Kven II Repository Instructions

Kven II is an experimental local AI architecture for persistent identity, memory, causal continuity, and autonomous existence across replaceable model and hardware components.

## Fixed locations

- Repository: `/opt/kven2`
- Python environment: `/opt/kven2/venv`
- Result root: `/home/eugene/kven-codex-results`
- Linked-worktree root: `/opt/kven2-worktrees`

Use English for code, comments, docstrings, tests, logs, configuration, prompts, and repository documentation. Before persistent changes, establish and verify a clean baseline. Keep the main worktree clean and make changes on a feature branch or in a linked worktree. Prefer small, testable changes. Run focused tests first, followed by the broader relevant suite.

Before completion, run `git diff --check`, report final Git status and changed files, and give explicit rollback instructions. Ordinary code, command, test, and commit confirmations are not required within an approved task. Stop only for architectural decisions, unexpected baseline conflicts, credential risk, protected-data access, production-system risk, API billing, or destructive operations outside task scope.

For a task intended for post-Codex integration, add `deployment-contract.json` using the schema documented in `docs/KVEN_INTEGRATION_WORKFLOW.md`. Declare changed paths, backups, migration checks, tests, services, readiness checks, acceptance, and rollback explicitly. Do not infer restart scope from filenames. The Codex runner embeds this contract in `result-manifest.json` and generates `result-summary.md` as the package entry point.

Known services are `agent-sandbox.service`, `kven2-main.service`, `kven2-telegram-gateway.service`, `kven-client-gateway.service`, and `email-agent.service`. Check service readiness only when a task affects runtime behavior.

Protect `/agent/data`: do not modify, migrate, reformat, structurally change, or use it for tests. Do not perform unrelated package, kernel, GPU-driver, CUDA, llama.cpp, model, or production model-configuration upgrades. Do not disclose secrets or read private environment values unless a task explicitly requires a specific safe check. Diagnostic and result directories intended for Eugene should normally be mode `0755`, and regular files mode `0644`.

The final report must be structured and include status, branch, baseline, commits, changed files, tests, service status, protected-data status, secret check, remaining actions, and rollback.
