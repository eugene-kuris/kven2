import unittest
from unittest.mock import patch

import agent_sandbox


class AgentSandboxTimeTests(unittest.TestCase):
    def test_get_time_exposes_unambiguous_timezone(self):
        with patch.object(
            agent_sandbox,
            "_system_timezone_name",
            return_value="Europe/Kyiv",
        ):
            result = agent_sandbox.get_time()

        self.assertEqual(
            result["timezone"],
            "Europe/Kyiv",
        )
        self.assertIn(
            "timezone_abbreviation",
            result,
        )
        self.assertIn(
            "utc_offset",
            result,
        )
        self.assertIn(
            "weekday",
            result,
        )

    def test_system_timezone_name_prefers_localtime_link(self):
        with (
            patch.dict(
                agent_sandbox.os.environ,
                {"TZ": "US/Eastern"},
                clear=False,
            ),
            patch.object(
                agent_sandbox.os.path,
                "realpath",
                return_value=(
                    "/usr/share/zoneinfo/Europe/Kyiv"
                ),
            ),
        ):
            self.assertEqual(
                agent_sandbox._system_timezone_name(),
                "Europe/Kyiv",
            )

    def test_hostile_tz_does_not_change_returned_zone(self):
        with (
            patch.dict(
                agent_sandbox.os.environ,
                {"TZ": "US/Eastern"},
                clear=False,
            ),
            patch.object(
                agent_sandbox,
                "_system_timezone_name",
                return_value="Europe/Kyiv",
            ),
        ):
            result = agent_sandbox.get_time()

        self.assertEqual(
            result["timezone"],
            "Europe/Kyiv",
        )
        self.assertEqual(
            result["timezone_abbreviation"],
            "EEST",
        )
        self.assertEqual(
            result["utc_offset"],
            "+0300",
        )


if __name__ == "__main__":
    unittest.main()
