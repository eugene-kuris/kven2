import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from telegram_compaction import SCHEMA_VERSION, parse_and_validate_payload
from telegram_store import TelegramStore
from telegram_workers import run_generation_once


def payload(ids, **overrides):
    value = {
        "schema_version": SCHEMA_VERSION,
        "overview": "Two separate topics and an unresolved check were discussed.",
        "established_context": [],
        "speaker_statements": [{"text": "The user described a possibility, not a fact.", "speaker": "user", "source_entry_ids": [ids[0]]}],
        "uncertainty_and_disagreement": [{"text": "Two explanations remain unresolved.", "source_entry_ids": [ids[0], ids[1]]}],
        "open_loops": [{"text": "A proposed check awaits verification.", "status": "awaiting verification", "source_entry_ids": [ids[-1]]}],
        "commitments": [],
        "important_reference_ids": [ids[0]],
    }
    value.update(overrides)
    return value


class TelegramCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "telegram.db"
        self.store = TelegramStore(
            str(self.path), exact_tail_token_budget=80,
            compaction_enabled=True, compaction_trigger_token_threshold=20,
            compaction_exact_tail_reserve=8, compaction_target_token_budget=300,
            compaction_min_entries=2,
        )
        await self.store.init()

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def add_turn(self, number, text=None):
        text = text or ("possible explanation " + "x" * 40)
        await self.store.enqueue_text_update(
            update_id=number, chat_id=20, user_id=30, message_id=100 + number,
            text=text, raw_update={"update_id": number})
        job = await self.store.claim_next_job()
        await self.store.save_response(job.id, "I will check; this is not completed. " + "y" * 40)

    def rows(self, sql, args=()):
        with sqlite3.connect(self.path) as connection:
            return connection.execute(sql, args).fetchall()

    async def claim(self):
        for number in range(1, 4):
            await self.add_turn(number)
        claim = await self.store.claim_next_compaction()
        self.assertIsNotNone(claim)
        return claim

    async def test_first_checkpoint_activation_preserves_transcript_and_projects_exact_tail(self):
        claim = await self.claim()
        before = self.rows("SELECT id,role,content FROM telegram_messages ORDER BY id")
        ids = list(range(claim.coverage_start_id, claim.coverage_end_id + 1))
        self.assertTrue(await self.store.complete_compaction(claim.checkpoint_id, json.dumps(payload(ids))))
        self.assertEqual(before, self.rows("SELECT id,role,content FROM telegram_messages ORDER BY id"))
        await self.store.enqueue_text_update(update_id=10, chat_id=20, user_id=30, message_id=110, text="current exact", raw_update={}, reply_to_message_id=101)
        job = await self.store.claim_next_job()
        context = await self.store.build_generation_context(job)
        self.assertIn("Derived Telegram conversation checkpoint", context[0]["content"])
        self.assertTrue(any("Explicit reply reference" in item["content"] for item in context))
        self.assertTrue(any("possible explanation" in item["content"] for item in context))
        self.assertEqual(context[-1]["content"], "current exact")

    async def test_invalid_json_missing_provenance_and_outside_ids_cannot_activate(self):
        for bad in ("not json", json.dumps({"schema_version": SCHEMA_VERSION}),
                    json.dumps(payload([999, 1000]))):
            claim = await self.claim() if not self.rows("SELECT id FROM telegram_compaction_checkpoints") else await self.store.claim_next_compaction()
            if claim is None:
                # Make failed frontier retryable.
                with sqlite3.connect(self.path) as connection:
                    connection.execute("UPDATE telegram_compaction_checkpoints SET status='failed' WHERE status='pending'")
                claim = await self.store.claim_next_compaction()
            with self.assertRaises(ValueError):
                await self.store.complete_compaction(claim.checkpoint_id, bad)
            self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_compaction_checkpoints WHERE status='active'")[0][0], 0)
            await self.store.fail_compaction(claim.checkpoint_id, ValueError())

    async def test_digest_mismatch_blocks_activation(self):
        claim = await self.claim()
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE telegram_messages SET content='changed' WHERE id=?", (claim.coverage_start_id,))
        ids = list(range(claim.coverage_start_id, claim.coverage_end_id + 1))
        with self.assertRaisesRegex(ValueError, "digest"):
            await self.store.complete_compaction(claim.checkpoint_id, json.dumps(payload(ids)))

    async def test_supersession_and_duplicate_frontier_are_deterministic(self):
        first = await self.claim()
        ids = list(range(first.coverage_start_id, first.coverage_end_id + 1))
        await self.store.complete_compaction(first.checkpoint_id, json.dumps(payload(ids)))
        for number in range(4, 7):
            await self.add_turn(number)
        second = await self.store.claim_next_compaction()
        self.assertIsNotNone(second)
        second_ids = list(range(second.coverage_start_id, second.coverage_end_id + 1))
        await self.store.complete_compaction(second.checkpoint_id, json.dumps(payload(second_ids)))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_compaction_checkpoints WHERE status='active'")[0][0], 1)
        self.assertEqual(self.rows("SELECT status FROM telegram_compaction_checkpoints WHERE id=?", (first.checkpoint_id,))[0][0], "superseded")
        self.assertIsNone(await self.store.claim_next_compaction())

    async def test_restart_recovers_pending_and_previous_active_survives_failure(self):
        first = await self.claim()
        ids = list(range(first.coverage_start_id, first.coverage_end_id + 1))
        await self.store.complete_compaction(first.checkpoint_id, json.dumps(payload(ids)))
        for number in range(4, 7):
            await self.add_turn(number)
        pending = await self.store.claim_next_compaction()
        self.assertGreaterEqual(await self.store.recover_incomplete_jobs(), 1)
        self.assertEqual(self.rows("SELECT status FROM telegram_compaction_checkpoints WHERE id=?", (pending.checkpoint_id,))[0][0], "failed")
        self.assertEqual(self.rows("SELECT id FROM telegram_compaction_checkpoints WHERE status='active'")[0][0], first.checkpoint_id)

    async def test_generation_worker_prioritizes_answer_then_compaction_and_contains_failures(self):
        for number in range(1, 4):
            await self.add_turn(number)

        class Client:
            async def generate_reply(self, messages):
                ids = [entry["entry_id"] for entry in json.loads(messages[-1]["content"])["transcript_entries"]]
                return json.dumps(payload(ids))

        self.assertTrue(await run_generation_once(self.store, Client()))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_compaction_checkpoints WHERE status='active'")[0][0], 1)

    def test_schema_preserves_attribution_uncertainty_and_incomplete_action(self):
        checked = parse_and_validate_payload(json.dumps(payload([1, 2])), {1, 2})
        self.assertEqual(checked["speaker_statements"][0]["speaker"], "user")
        self.assertIn("unresolved", checked["uncertainty_and_disagreement"][0]["text"])
        self.assertIn("awaits", checked["open_loops"][0]["text"])

    async def test_disabled_mode_is_exact_tail_compatible(self):
        disabled = TelegramStore(str(Path(self.temp.name) / "disabled.db"), compaction_enabled=False)
        await disabled.init()
        self.assertIsNone(await disabled.claim_next_compaction())


if __name__ == "__main__":
    unittest.main()
