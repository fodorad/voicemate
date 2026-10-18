import tempfile
import unittest

from tests.helpers import make_tool_context
from voicemate.agent.prompts import system_prompt
from voicemate.agent.tools import build_tools
from voicemate.config import Config


class TestAddressGuidance(unittest.TestCase):
    def test_an_address_from_the_user_means_fetch_page_and_it_asks_permission_itself(self):
        prompt = " ".join(system_prompt(Config()).split())
        self.assertIn("web address", prompt)
        self.assertIn("`fetch_page`", prompt)
        self.assertIn("asks Adam's permission itself", prompt)

    def test_the_model_is_told_never_to_claim_it_cannot_open_websites(self):
        prompt = " ".join(system_prompt(Config()).split())
        self.assertIn("never say you cannot open websites", prompt)

    def test_the_prompt_stays_static_across_calls(self):
        self.assertEqual(system_prompt(Config()), system_prompt(Config()))


class TestFetchPageDescription(unittest.TestCase):
    def test_it_covers_addresses_the_user_gives(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_tool_context(tmp)
            tools = {t.name: t for t in build_tools(ctx)}
            description = " ".join(tools["fetch_page"].description.split())
            ctx.notes.close()
        self.assertIn("address the user gives", description)
        self.assertIn("asks the user first", description)


if __name__ == "__main__":
    unittest.main()
