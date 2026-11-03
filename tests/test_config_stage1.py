import json
import os
from pathlib import Path
import tempfile
import unittest

from agent.config import DEFAULT_CONFIG_PATH, load_config
from agent.audit import write_audit_record


class StageOneConfigTests(unittest.TestCase):
    def test_default_config_has_required_sections_and_no_secret(self):
        config = load_config()
        self.assertEqual(config.section("vision")["model_family"], "RDV")
        self.assertIn("api_key_env", config.section("llm"))
        self.assertNotIn("api_key", config.section("llm"))
        self.assertTrue(DEFAULT_CONFIG_PATH.is_file())

    def test_partial_config_is_merged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "override.json"
            path.write_text(
                json.dumps({"video": {"sample_interval": 3.5}}),
                encoding="utf-8",
            )
            config = load_config(path)
            self.assertEqual(config.section("video")["sample_interval"], 3.5)
            self.assertEqual(config.section("vision")["model_family"], "RDV")


if __name__ == "__main__":
    unittest.main()

