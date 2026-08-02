import json
import tempfile
import unittest
from pathlib import Path

import hnswlib
import numpy as np

import hnsw


class HnswDynamicCapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory(
            prefix="kven2-hnsw-test-"
        )
        self.addCleanup(
            self.temp_directory.cleanup
        )

        self.original_state = {
            "index_path": hnsw.index_path,
            "id_map_path": hnsw.id_map_path,
            "initial_capacity": hnsw.INITIAL_CAPACITY,
            "hnsw_index": hnsw.hnsw_index,
            "id_to_hnsw": hnsw.id_to_hnsw,
            "hnsw_to_id": hnsw.hnsw_to_id,
            "next_hnsw_id": hnsw.next_hnsw_id,
            "save_counter": hnsw._SAVE_COUNTER,
        }
        self.addCleanup(self._restore_state)

        directory = Path(
            self.temp_directory.name
        )

        hnsw.index_path = str(
            directory / "index.bin"
        )
        hnsw.id_map_path = str(
            directory / "index.bin.id_map.json"
        )
        hnsw.INITIAL_CAPACITY = 4
        hnsw.hnsw_index = None
        hnsw.id_to_hnsw = {}
        hnsw.hnsw_to_id = {}
        hnsw.next_hnsw_id = 0
        hnsw._SAVE_COUNTER = 0

    def _restore_state(self):
        hnsw.index_path = self.original_state[
            "index_path"
        ]
        hnsw.id_map_path = self.original_state[
            "id_map_path"
        ]
        hnsw.INITIAL_CAPACITY = self.original_state[
            "initial_capacity"
        ]
        hnsw.hnsw_index = self.original_state[
            "hnsw_index"
        ]
        hnsw.id_to_hnsw = self.original_state[
            "id_to_hnsw"
        ]
        hnsw.hnsw_to_id = self.original_state[
            "hnsw_to_id"
        ]
        hnsw.next_hnsw_id = self.original_state[
            "next_hnsw_id"
        ]
        hnsw._SAVE_COUNTER = self.original_state[
            "save_counter"
        ]

    @staticmethod
    def _vectors(count):
        vectors = np.zeros(
            (count, hnsw.DIMENSION),
            dtype=np.float32,
        )

        for index in range(count):
            vectors[index, index] = 1.0

        return vectors

    def _create_empty_index(self):
        hnsw._reset_id_map(
            save_empty=True
        )
        hnsw._create_new_index(
            persist_empty=True
        )

    def test_empty_index_reloads_with_initial_capacity(self):
        self._create_empty_index()

        self.assertEqual(
            hnsw.hnsw_index.get_current_count(),
            0,
        )
        self.assertEqual(
            hnsw.hnsw_index.get_max_elements(),
            4,
        )

        hnsw.hnsw_index = None
        hnsw.id_to_hnsw = {}
        hnsw.hnsw_to_id = {}
        hnsw.next_hnsw_id = 0

        hnsw.init_hnsw()

        self.assertEqual(
            hnsw.hnsw_index.get_current_count(),
            0,
        )
        self.assertEqual(
            hnsw.hnsw_index.get_max_elements(),
            4,
        )
        self.assertEqual(
            hnsw.id_to_hnsw,
            {},
        )

    def test_capacity_grows_geometrically_before_add(self):
        hnsw.INITIAL_CAPACITY = 2
        self._create_empty_index()

        self.assertTrue(
            hnsw.add_to_hnsw(
                [1, 2],
                self._vectors(2),
            )
        )
        self.assertEqual(
            hnsw.hnsw_index.get_max_elements(),
            2,
        )

        self.assertTrue(
            hnsw.add_to_hnsw(
                [3],
                self._vectors(3)[2:3],
            )
        )
        self.assertEqual(
            hnsw.hnsw_index.get_max_elements(),
            4,
        )

        self.assertTrue(
            hnsw.add_to_hnsw(
                [4, 5],
                self._vectors(5)[3:5],
            )
        )
        self.assertEqual(
            hnsw.hnsw_index.get_current_count(),
            5,
        )
        self.assertEqual(
            hnsw.hnsw_index.get_max_elements(),
            8,
        )

        loaded = hnswlib.Index(
            space=hnsw.SPACE,
            dim=hnsw.DIMENSION,
        )
        loaded.load_index(
            hnsw.index_path
        )

        self.assertEqual(
            loaded.get_current_count(),
            5,
        )
        self.assertEqual(
            loaded.get_max_elements(),
            8,
        )

        with open(
            hnsw.id_map_path,
            "r",
            encoding="utf-8",
        ) as handle:
            id_map = json.load(handle)

        self.assertEqual(
            len(id_map),
            5,
        )

    def test_old_nonempty_index_is_expanded_without_rebuild(self):
        hnsw.INITIAL_CAPACITY = 2
        self._create_empty_index()

        vector = self._vectors(1)

        self.assertTrue(
            hnsw.add_to_hnsw(
                [101],
                vector,
            )
        )

        hnsw.hnsw_index = None
        hnsw.id_to_hnsw = {}
        hnsw.hnsw_to_id = {}
        hnsw.next_hnsw_id = 0
        hnsw.INITIAL_CAPACITY = 8

        hnsw.init_hnsw()

        self.assertEqual(
            hnsw.hnsw_index.get_current_count(),
            1,
        )
        self.assertEqual(
            hnsw.hnsw_index.get_max_elements(),
            8,
        )
        self.assertEqual(
            hnsw.id_to_hnsw,
            {101: 1},
        )

        neighbors = hnsw.get_nearest_neighbors(
            vector[0],
            k=1,
        )

        self.assertEqual(
            neighbors[0][0],
            101,
        )

        loaded = hnswlib.Index(
            space=hnsw.SPACE,
            dim=hnsw.DIMENSION,
        )
        loaded.load_index(
            hnsw.index_path
        )

        self.assertEqual(
            loaded.get_current_count(),
            1,
        )
        self.assertEqual(
            loaded.get_max_elements(),
            8,
        )


if __name__ == "__main__":
    unittest.main()
