# Telegram context compaction

The exact append-oriented `telegram_messages` transcript remains the source of
truth. A checkpoint is a derived, replaceable prompt projection over an explicit
contiguous range of stream entries. It never deletes transcript rows, writes to
semantic memory, changes owner identity, or establishes facts.

## Payload and epistemic rules

Schema `telegram-compaction-v1` contains a neutral overview, established
context, speaker-attributed statements, uncertainty/disagreement, open loops,
commitments, and important reference IDs. Every semantic item requires one or
more source entry IDs inside the checkpoint range. Assertions remain attributed;
contradictions remain unresolved; intentions and proposed fixes remain open
until a later cited message explicitly completes or cancels them. Unsupported
detail is omitted. Structural provenance limits invention but cannot prove that
model-generated wording is semantically faithful; human acceptance remains
required.

Checkpoints record stream and coverage IDs, a SHA-256 digest of exact role/text/
source-update records, schema version, model identifier when supplied, payload,
token estimate, prior/superseding IDs, validation state, bounded failure code,
and retry count. A partial unique index permits only one active and one pending
checkpoint per stream. Generation uses the exact full covered prefix rather than
recursive summary input, reducing summary-of-summary drift. Exact transcript
rebuild remains possible; semantic drift is still a known limitation.

## Lifecycle, scheduling, and fallback

After a stable assistant turn, the normal generation loop first claims any
waiting Telegram answer. Only when no queued or processing answer exists may it
claim durable compaction work. The prefix excludes the configured whole-entry
exact-tail reserve and therefore excludes the current batch and active
generation. The normal Kven scheduling path continues to preserve OWUI priority.
Validation and activation are one SQLite transaction. The previous active
checkpoint becomes `superseded` only when its replacement becomes `active`.

Invalid JSON/schema/provenance, digest drift, interruption, or model failure
cannot activate a checkpoint. The previous active checkpoint remains, or prompt
assembly falls back to the existing exact tail. Restart marks an interrupted
pending operation failed and it is safely retryable at the same frontier.
Failures never become transcript content and carry type/status only, not model
output or private text.

## Prompt and budgets

Prompt priority is current inbound batch, exact reply references, whole exact
recent entries, then the derived checkpoint. The checkpoint is explicitly
marked incomplete and subordinate. Open loops and commitments are retained
first within its budget; lower-priority derived sections are dropped whole.
Exact reply anchors outside both tail and coverage remain exact. Current input
is never trimmed, and logical transcript entries are never split. Accounting
reuses `estimate_tokens_from_chars` and is deterministic, content-free in
diagnostics, and intentionally not a model-tokenizer guarantee.

Configuration defaults to enabled with conservative thresholds; explicit
disabled mode preserves the prior behavior:

- `TELEGRAM_COMPACTION_ENABLED=1`
- `TELEGRAM_COMPACTION_TRIGGER_TOKEN_THRESHOLD=8192`
- `TELEGRAM_COMPACTION_EXACT_TAIL_RESERVE=4096`
- `TELEGRAM_COMPACTION_TARGET_TOKEN_BUDGET=1536`
- `TELEGRAM_COMPACTION_MIN_ENTRIES=4`

All numeric values must be positive; the enable flag accepts standard boolean
spellings. Compaction model calls expose no tools, so they cannot create tool or
memory side effects. No owner or model identifier is hard-coded.

## Operations, migration, and rollback

Use `scripts/telegram-compaction-status DATABASE` for identifier/count/status
diagnostics; it opens SQLite read-only and never prints transcript or summary
text. For a durable artifact, redirect its JSON output to a mode-0644 file under
`/home/eugene`. The integration workflow backs up the live database, applies
the idempotent schema to a disposable copy twice, checks integrity, stages the
exact feature SHA, and restarts only `kven2-telegram-gateway.service`.

To rebuild a suspect checkpoint, preserve its metadata, mark it rejected using
an independently reviewed maintenance action, and let the same exact frontier
be regenerated from transcript. Activation still requires normal validation.
There is deliberately no command that prints payloads or transcript text.

Rollback stops the gateway, restores the protected pre-stage SQLite backup
(including consistent WAL content through the integration workflow), restores
the prior Git state, and starts/checks the gateway. Do not reverse-migrate the
live database in place. Known limitations are structural rather than semantic
validation, estimator rather than backend tokenizer accounting, no topic
segmentation, and manual semantic acceptance for subtle unsupported inference.
