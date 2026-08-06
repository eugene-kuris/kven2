# Telegram owner conversation continuity

The Bot API gateway maps the authorized private owner to one durable relationship stream keyed by immutable numeric Telegram chat and sender IDs. The allowlist remains `TELEGRAM_ALLOWED_USER_ID`; usernames are not identity inputs and no additional users, groups, or channels are enabled.

## Transcript and prompt projection

Every accepted text or caption is appended exactly once to `telegram_messages`, with its update ID, Telegram message ID, receipt time, Telegram epoch time, reply target, stream, and batch association. One completed generation appends one assistant entry. Delivery chunks remain operational rows. Replies to the first or any later delivered chunk resolve through `telegram_delivery_chunks` to that single logical assistant entry.

The immutable transcript is not the prompt. At generation claim time the gateway projects the newest whole entries that fit `TELEGRAM_EXACT_TAIL_TOKEN_BUDGET` (default 4096 estimated tokens). This setting is the exact-transcript allocation before the request reaches the global Kven route, not the complete final message budget; it does not reserve system instructions, tools, memory, output, the current batch, or protected reply references. Oldest ordinary entries fall out first; entries and assistant answers are never split. The entire current inbound batch and exact messages explicitly referenced by its replies are always retained, even when they exceed the allocation. A referenced entry already in the tail is not duplicated.

Historical projected entries carry a deterministic Europe/Kyiv ISO timestamp and reply metadata. For every current inbound message, the transport metadata is a separate system context entry immediately before an exact, unmodified user entry. This keeps time and reply information visible without changing leading Telegram directive recognition. Internal tool traces are not transcript turns.

## Batching and recovery

`TELEGRAM_BATCH_DEBOUNCE_SECONDS` defaults to 1.5 seconds. Each arrival durably extends the pending batch and resets its ready time. Claiming atomically changes one batch to processing; later arrivals create the next batch, and a stream cannot claim another generation while one is processing. OWUI scheduling and the Kven HTTP/tool/memory path are unchanged.

Startup idempotently adds stream, metadata, ready-time, and job-message association structures to the existing gateway SQLite database. Existing jobs receive a stream and one-member batch. Valid original `date` and replied-to `message_id` values are defensively recovered from pre-feature `telegram_updates.raw_json` and copied to matching user transcript rows. Malformed, partial, mismatched, or unexpectedly typed JSON leaves metadata `NULL` and does not abort migration. Processing jobs return to queued; generated responses and chunk delivery state retain the existing recovery behavior.

## Deployment checklist

1. Stop the Telegram gateway and make a timestamped filesystem-level backup of the database plus WAL/SHM files.
2. Confirm ownership, free space, and backup readability; keep the backup outside the database directory.
3. Configure the two optional variables above, or accept their defaults.
4. Start one gateway instance; initialization performs the idempotent schema migration and legacy metadata backfill. Inspect identifier/count-only migration and recovery logs.
5. Run the manual checks: single message; rapid messages; replies to recent and tail-expired messages; a long time gap; restarts before generation, after generation, and during chunk delivery; no duplicate answer; one long logical answer; OWUI priority; directives; and reference to earlier dialogue.

Rollback requires stopping the gateway, preserving the post-migration database for investigation, restoring the complete pre-migration database/WAL/SHM backup, checking file ownership and permissions, deploying the prior code, and starting one gateway instance. SQLite cannot drop added columns safely in place, so do not attempt a reverse migration on production data.

Known limitations: no edit/delete reconciliation, media understanding, groups, multiple interlocutors, semantic summaries, topic boundaries, or neighboring reply-episode expansion. Missing reply targets remain unresolved. Tail accounting uses the repository's deterministic conservative character estimator rather than a model-specific tokenizer, and the Telegram allocation is not a hard cap for the complete final model request.
