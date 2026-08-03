import unittest

from kven2_profile import build_agent_profile_prompt


class AgentProfilePromptTests(unittest.TestCase):
    def test_empty_profile_produces_empty_prompt(self):
        self.assertEqual(build_agent_profile_prompt({}), "")

    def test_legacy_profile_output_is_preserved(self):
        profile = {
            "agent_name": "Kven II",
            "agent_role": "Research AI",
            "project_history": "Continuation of Kven I",
            "owner": "Eugene Kuris",
            "mission": "Assist in development",
        }

        expected = (
            "name: Kven II\n"
            "role: You are my friend.\n"
            "\n"
            "You are Kven II.\n"
            "Agent Role: Research AI\n"
            "Project History: Continuation of Kven I\n"
            "Owner: Eugene Kuris\n"
            "Mission: Assist in development\n"
            "\n"
        )

        self.assertEqual(build_agent_profile_prompt(profile), expected)

    def test_extended_fields_require_profile_version_two(self):
        profile = {
            "agent_name": "Kven II",
            "identity_instruction": "Maintain continuity.",
            "relationship": "Long-term collaboration",
        }

        prompt = build_agent_profile_prompt(profile)

        self.assertNotIn("AGENT PROFILE INSTRUCTIONS:", prompt)
        self.assertNotIn("Maintain continuity.", prompt)
        self.assertNotIn("Relationship:", prompt)

    def test_boolean_profile_version_does_not_enable_extended_fields(self):
        profile = {
            "profile_version": True,
            "identity_instruction": "Maintain continuity.",
        }

        prompt = build_agent_profile_prompt(profile)

        self.assertNotIn("AGENT PROFILE INSTRUCTIONS:", prompt)
        self.assertNotIn("Maintain continuity.", prompt)

    def test_extended_profile_uses_allowlisted_order(self):
        profile = {
            "profile_version": 2,
            "agent_name": "Kven II",
            "agent_role": "Research AI",
            "project_history": "Continuation of Kven I",
            "owner": "Eugene Kuris",
            "mission": "Develop cognitive architectures",
            "relationship": "Long-term collaboration",
            "communication_instruction": "Speak naturally.",
            "identity_instruction": "Maintain continuity.",
            "unknown_instruction": "This must not reach the prompt.",
        }

        prompt = build_agent_profile_prompt(profile)

        self.assertIn(
            "Relationship: Long-term collaboration",
            prompt,
        )
        self.assertIn(
            "IDENTITY AND CONTINUITY:\nMaintain continuity.",
            prompt,
        )
        self.assertIn(
            "COMMUNICATION:\nSpeak naturally.",
            prompt,
        )
        self.assertLess(
            prompt.index("IDENTITY AND CONTINUITY:"),
            prompt.index("COMMUNICATION:"),
        )
        self.assertNotIn("unknown_instruction", prompt)
        self.assertNotIn("This must not reach the prompt.", prompt)

    def test_non_string_instruction_is_ignored(self):
        profile = {
            "profile_version": 2,
            "agent_name": "Kven II",
            "identity_instruction": ["invalid"],
        }

        prompt = build_agent_profile_prompt(profile)

        self.assertNotIn("IDENTITY AND CONTINUITY:", prompt)
        self.assertNotIn("invalid", prompt)

    def test_missing_name_uses_stable_default(self):
        prompt = build_agent_profile_prompt(
            {
                "mission": "Test mission",
            }
        )

        self.assertTrue(
            prompt.startswith(
                "name: Kven II\n"
                "role: You are my friend.\n\n"
                "You are Kven II.\n"
            )
        )


if __name__ == "__main__":
    unittest.main()
