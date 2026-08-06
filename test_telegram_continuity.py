import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import routes
from telegram_store import TelegramStore
from tool_registry import export_openai_tools


class TelegramContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "telegram.db"
        self.store = TelegramStore(str(self.path), exact_tail_token_budget=80)
        await self.store.init()

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def enqueue(self, update_id, text, *, message_id=None, date=None, reply=None):
        return await self.store.enqueue_text_update(
            update_id=update_id,
            chat_id=200,
            user_id=300,
            message_id=message_id or update_id,
            text=text,
            raw_update={"update_id": update_id},
            message_date=date,
            reply_to_message_id=reply,
        )

    def rows(self, query, parameters=()):
        with sqlite3.connect(self.path) as connection:
            return connection.execute(query, parameters).fetchall()

    async def test_stream_is_reused_and_duplicate_is_append_idempotent(self):
        await self.enqueue(1, "one")
        await self.enqueue(2, "two")
        self.assertFalse(await self.enqueue(2, "two"))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_streams")[0][0], 1)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_messages")[0][0], 2)

    async def test_rapid_messages_form_one_ordered_durable_batch(self):
        await self.enqueue(1, "one")
        await self.enqueue(2, "two")
        job = await self.store.claim_next_job()
        self.assertEqual(job.batch_update_ids, (1, 2))
        self.assertEqual([m[0] for m in self.rows(
            "SELECT update_id FROM telegram_job_messages WHERE job_id=? ORDER BY ordinal", (job.id,)
        )], [1, 2])

    async def test_debounce_defaults_zero_and_resets_ready_time(self):
        self.assertEqual(TelegramStore(str(self.path)).batch_debounce_seconds, 0.0)
        delayed = TelegramStore(str(self.path), batch_debounce_seconds=5.0)
        with patch("telegram_store.time.time", return_value=100.0):
            await delayed.enqueue_text_update(
                update_id=20, chat_id=201, user_id=301, message_id=20,
                text="one", raw_update={"update_id": 20},
            )
        with patch("telegram_store.time.time", return_value=103.0):
            await delayed.enqueue_text_update(
                update_id=21, chat_id=201, user_id=301, message_id=21,
                text="two", raw_update={"update_id": 21},
            )
        ready_at = self.rows(
            "SELECT ready_at FROM telegram_jobs WHERE update_id=20"
        )[0][0]
        self.assertEqual(ready_at, 108.0)
        with patch("telegram_store.time.time", return_value=107.9):
            self.assertIsNone(await delayed.claim_next_job())
        with patch("telegram_store.time.time", return_value=108.0):
            self.assertEqual((await delayed.claim_next_job()).batch_update_ids, (20, 21))

    def test_invalid_store_configuration(self):
        with self.assertRaises(ValueError):
            TelegramStore(str(self.path), batch_debounce_seconds=-0.1)
        with self.assertRaises(ValueError):
            TelegramStore(str(self.path), exact_tail_token_budget=0)

    async def test_message_arriving_during_generation_is_next_batch(self):
        await self.enqueue(1, "first")
        active = await self.store.claim_next_job()
        await self.enqueue(2, "later")
        self.assertIsNone(await self.store.claim_next_job())
        await self.store.save_response(active.id, "answer")
        following = await self.store.claim_next_job()
        self.assertEqual(following.batch_update_ids, (2,))

    async def test_context_has_times_batch_boundaries_and_reply_reference(self):
        await self.enqueue(1, "old exact text", message_id=101, date=1_700_000_000)
        old_job = await self.store.claim_next_job()
        await self.store.save_response(old_job.id, "old answer")
        for number in range(2, 8):
            await self.enqueue(number, "x" * 90)
            job = await self.store.claim_next_job()
            await self.store.save_response(job.id, "y" * 90)
        await self.enqueue(8, "reply now", date=1_700_100_000, reply=101)
        job = await self.store.claim_next_job()
        context = await self.store.build_generation_context(job)
        joined = "\n".join(item["content"] for item in context)
        self.assertIn("old exact text", joined)
        self.assertIn("Explicit reply reference", joined)
        self.assertIn("2023-", joined)
        self.assertEqual(joined.count("old exact text"), 1)
        self.assertEqual(context[-1], {"role": "user", "content": "reply now"})
        self.assertIn("Replies to Telegram message 101", context[-2]["content"])

    async def test_generation_context_is_deterministic(self):
        await self.enqueue(1, "same", date=1_700_000_000)
        job = await self.store.claim_next_job()
        self.assertEqual(
            await self.store.build_generation_context(job),
            await self.store.build_generation_context(job),
        )

    async def test_current_batch_is_never_trimmed_and_answers_are_whole(self):
        await self.enqueue(1, "historical")
        first = await self.store.claim_next_job()
        answer = "A" * 500
        await self.store.save_response(first.id, answer)
        await self.enqueue(2, "B" * 500)
        current = await self.store.claim_next_job()
        context = await self.store.build_generation_context(current)
        self.assertEqual(context[-1]["content"], "B" * 500)
        contents = [item["content"] for item in context]
        self.assertNotIn(answer, "\n".join(contents))

    async def test_restart_migrates_actual_legacy_schema_idempotently(self):
        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "legacy.db"
        with sqlite3.connect(self.path) as connection:
            connection.executescript("""
                CREATE TABLE telegram_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE telegram_updates (
                    update_id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                    text TEXT NOT NULL, raw_json TEXT NOT NULL,
                    received_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z');
                CREATE TABLE telegram_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, update_id INTEGER NOT NULL UNIQUE,
                    chat_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL, text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0, response_text TEXT,
                    telegram_response_message_id INTEGER, last_error TEXT,
                    created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
                    updated_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
                    started_at TEXT, delivery_started_at TEXT, completed_at TEXT);
                CREATE TABLE telegram_delivery_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                    telegram_message_id INTEGER, created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
                    updated_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
                    started_at TEXT, delivered_at TEXT, UNIQUE(job_id, chunk_index));
                CREATE TABLE telegram_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL,
                    source_update_id INTEGER NOT NULL, telegram_message_id INTEGER,
                    created_at TEXT NOT NULL DEFAULT '2026-01-01T00:00:00Z',
                    UNIQUE(source_update_id, role));
            """)
            states = ["queued", "processing", "responded", "sending", "delivered"]
            for number, status in enumerate(states, 1):
                raw = ('{"update_id":%d,"message":{"message_id":%d,"date":17000000%d,'
                       '"reply_to_message":{"message_id":77},"text":"m%d"}}') % (
                           number, 100 + number, number, number
                       )
                connection.execute(
                    "INSERT INTO telegram_updates(update_id,chat_id,user_id,message_id,text,raw_json) VALUES(?,?,?,?,?,?)",
                    (number, 200, 300, 100 + number, f"m{number}", raw),
                )
                response = None if status in {"queued", "processing"} else f"a{number}"
                connection.execute(
                    "INSERT INTO telegram_jobs(update_id,chat_id,user_id,message_id,text,status,response_text,attempts,delivery_attempts) VALUES(?,?,?,?,?,?,?,?,?)",
                    (number, 200, 300, 100 + number, f"m{number}", status, response, 1, 1),
                )
                connection.execute(
                    "INSERT INTO telegram_messages(chat_id,role,content,source_update_id,telegram_message_id) VALUES(?,?,?,?,?)",
                    (200, "user", f"m{number}", number, 100 + number),
                )
                if response is not None:
                    connection.execute(
                        "INSERT INTO telegram_messages(chat_id,role,content,source_update_id,telegram_message_id) VALUES(?,?,?,?,?)",
                        (200, "assistant", response, number, 500 + number),
                    )
            sending_job = connection.execute("SELECT id FROM telegram_jobs WHERE update_id=4").fetchone()[0]
            connection.execute(
                "INSERT INTO telegram_delivery_chunks(job_id,chunk_index,text,status,attempts,telegram_message_id) VALUES(?,?,?,?,?,?)",
                (sending_job, 0, "a4", "sending", 1, None),
            )
            connection.execute(
                "INSERT INTO telegram_updates(update_id,chat_id,user_id,message_id,text,raw_json) VALUES(99,200,300,199,'bad','not json')"
            )
            connection.execute(
                "INSERT INTO telegram_jobs(update_id,chat_id,user_id,message_id,text,status) VALUES(99,200,300,199,'bad','queued')"
            )
            connection.execute(
                "INSERT INTO telegram_messages(chat_id,role,content,source_update_id,telegram_message_id) VALUES(200,'user','bad',99,199)"
            )
        restarted = TelegramStore(str(self.path))
        await restarted.init()
        snapshot = self.rows(
            "SELECT update_id,message_date,reply_to_message_id FROM telegram_updates ORDER BY update_id"
        )
        await restarted.init()
        self.assertEqual(snapshot, self.rows(
            "SELECT update_id,message_date,reply_to_message_id FROM telegram_updates ORDER BY update_id"
        ))
        self.assertEqual(snapshot[:5], [(n, 170_000_000 + n, 77) for n in range(1, 6)])
        self.assertEqual(snapshot[-1], (99, None, None))
        self.assertEqual(self.rows(
            "SELECT telegram_date,reply_to_message_id FROM telegram_messages WHERE role='user' AND source_update_id=1"
        ), [(170_000_001, 77)])
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_jobs")[0][0], 6)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_delivery_chunks")[0][0], 1)

    async def test_long_delivery_is_one_assistant_transcript_entry(self):
        await self.enqueue(1, "question")
        job = await self.store.claim_next_job()
        await self.store.save_response(job.id, "z" * 9000)
        while True:
            delivery = await self.store.claim_next_delivery()
            if delivery is None:
                break
            await self.store.mark_delivery_chunk_delivered(delivery.chunk_id, 500 + delivery.chunk_index)
        self.assertGreater(self.rows("SELECT COUNT(*) FROM telegram_delivery_chunks")[0][0], 1)
        self.assertEqual(self.rows("SELECT COUNT(*) FROM telegram_messages WHERE role='assistant'")[0][0], 1)

    async def test_reply_to_every_delivery_chunk_resolves_one_logical_answer(self):
        await self.enqueue(1, "question", message_id=101)
        original = await self.store.claim_next_job()
        answer = "unique logical answer " + ("z" * 9000)
        await self.store.save_response(original.id, answer)
        delivered_ids = []
        while delivery := await self.store.claim_next_delivery():
            telegram_id = 700 + delivery.chunk_index
            delivered_ids.append(telegram_id)
            await self.store.mark_delivery_chunk_delivered(delivery.chunk_id, telegram_id)
        self.assertGreater(len(delivered_ids), 1)
        for offset, reply_id in enumerate((delivered_ids[0], delivered_ids[-1]), 2):
            await self.enqueue(offset, f"reply {offset}", reply=reply_id)
            current = await self.store.claim_next_job()
            context = await self.store.build_generation_context(current)
            joined = "\n".join(item["content"] for item in context)
            self.assertEqual(joined.count(answer), 1)
            await self.store.save_response(current.id, f"followup {offset}")
        self.assertEqual(self.rows(
            "SELECT COUNT(*) FROM telegram_messages WHERE role='assistant' AND source_update_id=1"
        )[0][0], 1)

    async def test_directives_survive_actual_route_preparation_with_metadata(self):
        cases = {
            "#tools": "help", "#tools fetch_url": "help",
            "#get_time": "tool", "#read_file /tmp/example": "tool",
            "#fetch_url https://example.invalid": "tool",
            "#web_search deterministic query": "tool",
        }
        for number, (text, expected) in enumerate(cases.items(), 30):
            await self.enqueue(number, text, date=1_700_000_000, reply=101)
            job = await self.store.claim_next_job()
            context = await self.store.build_generation_context(job)
            self.assertEqual(context[-1]["content"], text)
            self.assertIn("Telegram transport metadata", context[-2]["content"])
            result = routes._prepare_explicit_tool_directive({
                "model": "test", "messages": context,
                "tools": export_openai_tools(), "tool_choice": "auto",
            })
            self.assertEqual(result["kind"], expected, text)
            await self.store.save_response(job.id, "done")
        await self.enqueue(50, "ordinary text with #web_search inside")
        job = await self.store.claim_next_job()
        context = await self.store.build_generation_context(job)
        result = routes._prepare_explicit_tool_directive({
            "messages": context, "tools": export_openai_tools(), "tool_choice": "auto"
        })
        self.assertEqual(result["kind"], "none")

    async def test_store_closes_every_sqlite_connection(self):
        real_connect = sqlite3.connect
        counts = {"opened": 0, "closed": 0}

        class TrackingConnection(sqlite3.Connection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._tracking_closed = False
                counts["opened"] += 1

            def close(self):
                if not self._tracking_closed:
                    self._tracking_closed = True
                    counts["closed"] += 1
                return super().close()

        def tracking_connect(*args, **kwargs):
            kwargs["factory"] = TrackingConnection
            return real_connect(*args, **kwargs)

        tracked_path = Path(self.temporary.name) / "tracked.db"

        with patch(
            "telegram_store.sqlite3.connect",
            side_effect=tracking_connect,
        ):
            store = TelegramStore(str(tracked_path))

            await store.init()
            await store.advance_update_offset(10)
            self.assertEqual(
                await store.get_next_update_offset(),
                10,
            )

            accepted = await store.enqueue_text_update(
                update_id=1000,
                chat_id=200,
                user_id=300,
                message_id=1000,
                text="connection lifecycle",
                raw_update={"update_id": 1000},
            )
            self.assertTrue(accepted)

            job = await store.claim_next_job()
            self.assertIsNotNone(job)

            await store.build_generation_context(job)
            await store.save_response(job.id, "answer")

            delivery = await store.claim_next_delivery()
            self.assertIsNotNone(delivery)

            await store.mark_delivery_chunk_delivered(
                delivery.chunk_id,
                2000,
            )

            await store.load_conversation(200)
            await store.recover_incomplete_jobs()

        self.assertGreater(counts["opened"], 0)
        self.assertEqual(
            counts["opened"],
            counts["closed"],
        )


if __name__ == "__main__":
    unittest.main()
