import asyncio
import base64
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from telegram_bot_api import TelegramBotApi, TelegramBotApiError, TelegramFile
from telegram_store import TelegramStore
from telegram_kven_client import TelegramKvenClient
from telegram_updates import TelegramImageMedia, TelegramUpdateError, parse_authorized_text_update
from telegram_workers import run_polling_once


def image_update(*, update_id=10, message_id=20, caption=None, document=None):
    message = {
        "message_id": message_id,
        "date": 1_700_000_000 + update_id,
        "chat": {"id": 30, "type": "private"},
        "from": {"id": 40, "is_bot": False, "first_name": "Owner"},
        "reply_to_message": {"message_id": 7},
    }
    if caption is not None:
        message["caption"] = caption
    if document is None:
        message["photo"] = [
            {"file_id": "small", "file_unique_id": "same", "width": 90, "height": 90, "file_size": 50},
            {"file_id": "large", "file_unique_id": "same", "width": 900, "height": 700, "file_size": 100},
        ]
    else:
        message["document"] = document
    return {"update_id": update_id, "message": message}


class UpdateParsingTests(unittest.TestCase):
    def test_photo_selects_largest_and_preserves_metadata(self):
        parsed = parse_authorized_text_update(
            image_update(caption="Read this"), allowed_user_id=40
        )
        self.assertEqual(parsed.text, "Read this")
        self.assertEqual(parsed.message_date, 1_700_000_010)
        self.assertEqual(parsed.reply_to_message_id, 7)
        self.assertEqual((parsed.update_id, parsed.message_id, parsed.user_id, parsed.chat_id), (10, 20, 40, 30))
        self.assertEqual(parsed.media.file_id, "large")
        self.assertEqual((parsed.media.file_unique_id, parsed.media.mime_type), ("same", "image/jpeg"))
        self.assertEqual((parsed.media.width, parsed.media.height, parsed.media.file_size), (900, 700, 100))

    def test_captionless_photo_is_valid_neutral_input(self):
        parsed = parse_authorized_text_update(image_update(), allowed_user_id=40)
        self.assertEqual(parsed.text, "Image attachment.")
        self.assertIsNotNone(parsed.media)

    def test_image_document_is_accepted_and_non_image_is_not_visual(self):
        parsed = parse_authorized_text_update(image_update(document={
            "file_id": "doc", "file_unique_id": "uniq", "mime_type": "image/png",
            "file_name": "scan.png", "file_size": 123,
        }), allowed_user_id=40)
        self.assertEqual(parsed.media.kind, "document")
        self.assertEqual((parsed.media.filename, parsed.media.mime_type), ("scan.png", "image/png"))
        unsupported = image_update(document={
            "file_id": "pdf", "file_unique_id": "pdf-u", "mime_type": "application/pdf",
        })
        unsupported["message"]["caption"] = "Please inspect this PDF"
        self.assertIsNone(parse_authorized_text_update(unsupported, allowed_user_id=40))

    def test_malformed_image_metadata_fails_explicitly(self):
        malformed = image_update()
        malformed["message"]["photo"][1].pop("file_id")
        with self.assertRaises(TelegramUpdateError):
            parse_authorized_text_update(malformed, allowed_user_id=40)
        with self.assertRaises(TelegramUpdateError):
            parse_authorized_text_update(image_update(document={
                "file_id": "doc", "mime_type": "image/png",
            }), allowed_user_id=40)


class Response:
    def __init__(self, payload=None, *, content=b"", headers=None, status=200):
        self.payload = payload
        self.content = content
        self.headers = headers or {}
        self.status = status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class Http:
    def __init__(self, post_result=None, get_result=None):
        self.post_result = post_result
        self.get_result = get_result
        self.calls = []

    async def post(self, url, *, json, timeout):
        self.calls.append(("post", url, json, timeout))
        return self.post_result

    async def get(self, url, *, timeout):
        self.calls.append(("get", url, None, timeout))
        return self.get_result


class KvenHttp:
    def __init__(self):
        self.payload = None

    async def post(self, url, *, json, timeout):
        self.payload = json
        return Response({
            "choices": [{"message": {"role": "assistant", "content": "Seen"}, "finish_reason": "stop"}]
        })


class BotApiMediaTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_file_shape_and_download_byte_identity(self):
        content = b"\x89PNG\r\nfixture"
        http = Http(
            Response({"ok": True, "result": {
                "file_id": "f1", "file_unique_id": "u1", "file_path": "photos/a.png", "file_size": len(content),
            }}),
            Response(content=content, headers={"content-length": str(len(content))}),
        )
        api = TelegramBotApi("TOKEN", client=http)
        file = await api.get_file("f1")
        downloaded = await api.download_file(file.file_path, max_bytes=100)
        self.assertEqual(downloaded, content)
        self.assertEqual(http.calls[0][2], {"file_id": "f1"})
        self.assertTrue(http.calls[0][1].endswith("/botTOKEN/getFile"))
        self.assertTrue(http.calls[1][1].endswith("/file/botTOKEN/photos/a.png"))

    async def test_malformed_and_oversize_errors_do_not_expose_token(self):
        credential_fixture = "PRIVATE_TEST_TOKEN"
        api = TelegramBotApi(credential_fixture, client=Http(Response({"ok": True, "result": {"file_id": "f1"}})))
        with self.assertRaises(TelegramBotApiError) as malformed:
            await api.get_file("f1")
        self.assertNotIn(credential_fixture, str(malformed.exception))
        api = TelegramBotApi(credential_fixture, client=Http(get_result=Response(content=b"abc", headers={"content-length": "999"})))
        with self.assertRaises(TelegramBotApiError) as oversized:
            await api.download_file("photos/a.png", max_bytes=3)
        self.assertNotIn(credential_fixture, str(oversized.exception))


class Bot:
    def __init__(self, content):
        self.content = content
        self.downloads = 0

    async def get_updates(self, *, offset, timeout):
        return []

    async def get_file(self, file_id):
        return TelegramFile(file_id, "same", "photos/image.jpg", len(self.content))

    async def download_file(self, file_path, *, max_bytes):
        self.downloads += 1
        return self.content


class DurableMediaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "gateway.db"
        self.store = TelegramStore(str(self.db))
        await self.store.init()

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def enqueue_media(self, update_id=10, message_id=20, text="Read this"):
        return await self.store.enqueue_text_update(
            update_id=update_id, chat_id=30, user_id=40, message_id=message_id,
            text=text, raw_update=image_update(update_id=update_id, message_id=message_id, caption=text),
            message_date=1_700_000_000 + update_id, reply_to_message_id=7,
            media=TelegramImageMedia("photo", "large", "same", "image/jpeg", width=900, height=700, file_size=12),
        )

    async def test_pending_media_survives_restart_and_blocks_generation(self):
        await self.enqueue_media()
        self.assertIsNone(await self.store.claim_next_job())
        reopened = TelegramStore(str(self.db))
        await reopened.init()
        pending = await reopened.get_pending_media()
        self.assertEqual((pending["update_id"], pending["file_id"]), (10, "large"))
        self.assertEqual(await reopened.recover_incomplete_jobs(), 0)

    async def test_download_is_durable_idempotent_and_actual_bytes_reach_context(self):
        fixture = b"vision-bytes"
        await self.enqueue_media()
        bot = Bot(fixture)
        await run_polling_once(self.store, bot, allowed_user_id=40, timeout=1)
        job = await self.store.claim_next_job()
        context = await self.store.build_generation_context(job)
        current = context[-1]["content"]
        self.assertEqual(current[0], {"type": "text", "text": "Read this"})
        data_url = current[1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(data_url.split(",", 1)[1]), fixture)
        http = KvenHttp()
        client = TelegramKvenClient(model="test", client=http)
        self.assertEqual(await client.generate_reply(context), "Seen")
        request_url = http.payload["messages"][-1]["content"][1]["image_url"]["url"]
        self.assertEqual(base64.b64decode(request_url.split(",", 1)[1]), fixture)
        with sqlite3.connect(self.db) as connection:
            row = connection.execute("SELECT status,content_sha256,local_path FROM telegram_media WHERE update_id=10").fetchone()
        self.assertEqual(row[0], "ready")
        self.assertTrue((self.db.parent / "telegram_media" / row[2]).is_file())
        self.assertFalse(await self.enqueue_media())
        self.assertIsNone(await self.store.get_pending_media())

    async def test_mixed_batch_order_and_reply_metadata_are_preserved(self):
        await self.store.enqueue_text_update(
            update_id=9, chat_id=30, user_id=40, message_id=19, text="before",
            raw_update={"update_id": 9}, message_date=1_700_000_009,
        )
        await self.enqueue_media()
        await self.store.complete_media(10, "photos/image.jpg", b"vision-bytes")
        job = await self.store.claim_next_job()
        self.assertEqual(job.batch_update_ids, (9, 10))
        context = await self.store.build_generation_context(job)
        users = [item for item in context if item["role"] == "user"]
        self.assertEqual(users[0]["content"], "before")
        self.assertIsInstance(users[1]["content"], list)
        self.assertIn("Replies to Telegram message 7", context[-2]["content"])


class MigrationValidatorTests(unittest.TestCase):
    def test_smoke_is_independent_of_existing_queued_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db = Path(temp_dir) / "gateway.db"

            async def seed_unrelated_job():
                store = TelegramStore(str(db))
                await store.init()
                inserted = await store.enqueue_text_update(
                    update_id=8_100_000_001,
                    chat_id=8_100_000_002,
                    user_id=8_100_000_003,
                    message_id=8_100_000_004,
                    text="unrelated queued job",
                    raw_update={"update_id": 8_100_000_001},
                )
                self.assertTrue(inserted)

            asyncio.run(seed_unrelated_job())
            env = os.environ.copy()
            env["TELEGRAM_MIGRATION_DB"] = str(db)
            repository = Path(__file__).resolve().parent
            completed = subprocess.run(
                [
                    sys.executable,
                    str(repository / "scripts" / "validate-telegram-vision-migration"),
                    "smoke",
                ],
                cwd=repository,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=f"stdout={completed.stdout!r} stderr={completed.stderr!r}",
            )

            async def verify_unrelated_job_remains():
                store = TelegramStore(str(db))
                await store.init()
                job = await store.claim_next_job()
                self.assertIsNotNone(job)
                self.assertEqual(job.batch_update_ids, (8_100_000_001,))

            asyncio.run(verify_unrelated_job_remains())


if __name__ == "__main__":
    unittest.main()
