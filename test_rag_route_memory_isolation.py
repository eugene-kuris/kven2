import ast
import unittest
from pathlib import Path


class RagRouteMemoryIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(
            "/opt/kven2/routes.py"
        ).read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_unconditional_semantic_context_is_not_called(self):
        calls = []

        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue

            function = node.func

            if (
                isinstance(function, ast.Name)
                and function.id == "get_semantic_context"
            ):
                calls.append(node.lineno)

            if (
                isinstance(function, ast.Attribute)
                and function.attr == "get_semantic_context"
            ):
                calls.append(node.lineno)

        self.assertEqual(calls, [])

    def test_query_aware_retrieval_remains_active(self):
        calls = []

        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue

            function = node.func

            if (
                isinstance(function, ast.Name)
                and function.id == "retrieve_context"
            ):
                calls.append(node.lineno)

        self.assertTrue(calls)

    def test_legacy_semantic_prompt_marker_is_absent(self):
        self.assertNotIn(
            "--- SEMANTIC MEMORY (Learned Knowledge) ---",
            self.source,
        )

    def test_vector_retrieval_prompt_marker_remains(self):
        self.assertIn(
            "VECTOR RETRIEVAL CONTEXT:",
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
