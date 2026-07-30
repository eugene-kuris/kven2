import atexit
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hnsw

# Unit-test processes must not retain the production persistence handler.
atexit.unregister(hnsw.save_hnsw)


class FakeIndex:
    def __init__(self, payload=b"saved-index"):
        self.payload = payload
        self.saved_path = None

    def save_index(self, path):
        self.saved_path = path
        Path(path).write_bytes(self.payload)


class HnswShutdownTests(unittest.TestCase):
    def test_uninitialized_index_does_not_touch_existing_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "hnsw_index.bin"
            map_path = root / "hnsw_index.bin.id_map.json"

            index_before = b"production-index-sentinel"
            map_before = b'{"33": 33}'

            index_path.write_bytes(index_before)
            map_path.write_bytes(map_before)

            with patch.object(hnsw, "index_path", str(index_path)), \
                 patch.object(hnsw, "id_map_path", str(map_path)), \
                 patch.object(hnsw, "hnsw_index", None), \
                 patch.object(hnsw, "id_to_hnsw", {}):
                result = hnsw.save_hnsw()

            self.assertFalse(result)
            self.assertEqual(index_path.read_bytes(), index_before)
            self.assertEqual(map_path.read_bytes(), map_before)

    def test_initialized_index_still_saves_index_and_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "hnsw_index.bin"
            map_path = root / "hnsw_index.bin.id_map.json"
            fake_index = FakeIndex()

            with patch.object(hnsw, "index_path", str(index_path)), \
                 patch.object(hnsw, "id_map_path", str(map_path)), \
                 patch.object(hnsw, "hnsw_index", fake_index), \
                 patch.object(hnsw, "id_to_hnsw", {7: 1}), \
                 patch.object(hnsw, "_SAVE_COUNTER", 4):
                result = hnsw.save_hnsw()

            self.assertTrue(result)
            self.assertEqual(index_path.read_bytes(), b"saved-index")
            self.assertEqual(
                json.loads(map_path.read_text(encoding="utf-8")),
                {"7": 1},
            )


if __name__ == "__main__":
    unittest.main()
