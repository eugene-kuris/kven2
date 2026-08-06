# Codex CLI Task Workflow

## Durable execution is the default

Run long Codex and operational tasks in a named GNU screen session so execution
survives the initiating SSH connection. Create a logged session with:

```bash
screen -L -Logfile /home/eugene/TASK-screen.log -S task-name
```

Detach with `Ctrl-a d`, list sessions with `screen -ls`, and reattach with
`screen -r task-name`. The durable log is an operational evidence artifact and
must use a stable absolute path. When the command finishes, screen normally
removes the finished session; `screen -ls` will no longer list it, while the log
remains available. Result metadata must record the manager, session name,
reattach command, durable log, durable runner/result paths, and whether execution
was independent of the initiating SSH connection.

## Purpose and trust boundary

`scripts/kven-codex-task` turns an approved task prompt into an isolated, durable Codex CLI run. It verifies a clean local `main`, creates a unique feature branch and linked worktree, runs Codex there, and preserves both the work and an inspection package. The runner trusts the installed Codex CLI and the existing repository, but it does not grant access to secrets or protected runtime data.

Authentication uses the existing ChatGPT-plan CLI login. The runner executes `codex login status`; it neither reads nor copies `auth.json`, API keys, or token values. Plus and other plan usage limits still apply.

## Safety model

Project `.codex/config.toml` gives direct sessions safe defaults: approval policy `never`, read-only sandboxing, and disabled web search. It does not select a model or provider.

The runner intentionally overrides sandboxing with explicit `--sandbox danger-full-access --ask-for-approval never`. This is necessary because the agent must edit and commit inside its dedicated linked worktree. The isolation boundary is the pre-created feature worktree plus the execution contract, not the direct-session default. Network access is prohibited by the default contract. `--allow-network` permits it only when the supplied task explicitly requires it.

Never use the runner for tasks that require reading private environment values, credentials, databases, private logs, or `/agent/data`. The runner records only `/agent/data` kernel mount metadata and does not traverse or read the protected data. It never merges, pushes, deploys, restarts services, or removes the feature worktree automatically.

## Invocation

```bash
scripts/kven-codex-task task.md
scripts/kven-codex-task - < task.md
scripts/kven-codex-task --model MODEL task.md
scripts/kven-codex-task --allow-network task.md
scripts/kven-codex-task --help
```

Use `--model` only for an intentional model choice. `--allow-network` does not create a general network allowance; the task itself must identify the required network activity.

## Task metadata

Put an exact, unindented line such as `TASK ID: WORK-KVEN-123` in the prompt. If absent, the input filename stem becomes the task ID; stdin falls back to `stdin-task`. The runner sanitizes the value for branch and directory names. Empty prompts are rejected.

Before creating anything, the runner requires Codex, successful ChatGPT authentication, `/opt/kven2` checked out on `main`, a clean main worktree, and equality between main HEAD and the local `origin/main` ref. It does not fetch; this comparison is intentionally local.

## Results

Packages are created below `/home/eugene/kven-codex-results/<task>-<timestamp>-<pid>/`. Directories use mode `0755` and regular files use `0644`. A package includes:

- `result-manifest.json`, the canonical machine-readable result (schema `2.0`);
- `result-summary.md`, the canonical human entry point and next command;

- defensively redacted original/effective prompts, task ID, timestamps, duration, and summary;
- Codex version and credential-free authentication status;
- final response, stdout, progress/stderr, and exit code;
- main baseline, branch/worktree details, final Git state, commits, changed files, and diff summary;
- post-run service states, protected mount metadata, and an obvious-secret-pattern scan.

The package deliberately excludes authentication files, private `.env` contents, tokens, credentials, databases, and private runtime logs. Interrupted and failed runs preserve whatever evidence was produced.

When a committed `deployment-contract.json` is present, the runner validates it before embedding it. Literal credential-bearing argv or private-key material is refused without persisting the raw contract. A malformed/unsafe contract is recorded as a redacted error rather than interpreted from the final natural-language response. The contract must declare non-empty `result_validation_tests`. The runner snapshots exact feature branch/HEAD/index/status, executes the commands, and requires the snapshot to remain identical after each command. It records exact names/redacted argv in contract order, timestamps, duration, exit code, pass/fail, bounded output, and package-relative artifacts with sizes and SHA-256 values. Empty, mismatched, failed, tampered, or repository-mutating evidence forces a failed final status.

The canonical manifest records `actual_runtime_model`, `actual_runtime_provider`, `actual_reasoning_effort`, `runtime_session_id`, `token_usage`, and `codex_runtime_seconds` directly. Startup facts come only from the delimited Codex stderr header, token usage only from the terminal stderr usage record, and duration from the runner monotonic timer (also written to `duration-seconds.txt`). Requested and actual models remain distinct; a mismatch forces failure. If an older outer runner is bootstrapping an unmerged runner feature, postprocessing must set `evidence_provenance.method` to `bootstrap_postprocessing`, state that the executing runner lacked the feature, source session/model/provider/reasoning only from the trusted startup header, token usage only after the terminal record exists, duration only from `duration-seconds.txt`, record the execution transport, and rescan the final evidence tree. Unavailable values remain null; they are never estimated.

Post-Codex integration is described in `KVEN_INTEGRATION_WORKFLOW.md`. Begin with the exact command printed in `result-summary.md`.

## Branch and worktree lifecycle

Successful creation produces `codex/<task>-<timestamp>-<pid>` and `/opt/kven2-worktrees/<task>-<timestamp>-<pid>`. Both remain after completion.

Inspect:

```bash
git -C /opt/kven2-worktrees/TASK-RUN status --short --branch
git -C /opt/kven2-worktrees/TASK-RUN log --oneline main..HEAD
git -C /opt/kven2-worktrees/TASK-RUN diff main...HEAD
```

Continue by opening the preserved worktree and committing further coherent changes. To merge after review, return to `/opt/kven2`, verify it is clean and current, then run `git merge --no-ff codex/TASK-RUN`. Deployment and service operations are separate, explicitly approved procedures.

To roll back an unmerged run, first archive anything needed, then from the main worktree run:

```bash
git -C /opt/kven2 worktree remove /opt/kven2-worktrees/TASK-RUN
git -C /opt/kven2 branch -D codex/TASK-RUN
```

These cleanup commands are destructive and are never run automatically. If already merged, use `git revert <merge-or-commit>` rather than rewriting shared history.

## Troubleshooting

- **Authentication:** run `codex login status`. If logged out, complete normal ChatGPT CLI login outside the runner; never put an API key in the task.
- **Dirty main:** inspect `git -C /opt/kven2 status`. Commit, move, or deliberately discard changes through the normal repository workflow.
- **Stale `origin/main`:** the runner never fetches. Update the local ref through an approved offline/online Git workflow, then recheck equality with main HEAD.
- **Codex failure:** inspect `stderr-progress.txt`, `stdout.txt`, `final-git-state.txt`, and `summary.txt`; continue in the preserved worktree if safe.
- **Interrupted execution:** inspect the printed or partially created result directory and preserved worktree. Confirm no Codex process remains before retrying.
- **Plus usage limits:** wait for the plan limit to reset or use an approved plan/model choice. Do not substitute an API key.
