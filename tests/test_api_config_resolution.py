from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import llm_client
import deai_api


LLM_ENV_VARS = (
    "MINIMAX_API_KEY",
    "MINIMAX_MODEL",
    "MINIMAX_BASE_URL",
    "MINIMAX_API_TYPE",
    "BAIBAIAIGC_API_KEY",
    "BAIBAIAIGC_MODEL",
    "BAIBAIAIGC_BASE_URL",
    "BAIBAIAIGC_API_TYPE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


def _clear_env(env: dict[str, str]) -> dict[str, str]:
    cleaned = dict(env)
    for name in LLM_ENV_VARS:
        cleaned.pop(name, None)
    return cleaned


class ReadApiConfigEnvAliasTests(unittest.TestCase):
    """:func:`llm_client.read_api_config` resolves env var aliases."""

    def test_minimax_env_aliases_resolve(self) -> None:
        env = _clear_env(os.environ)
        env.update(
            MINIMAX_API_KEY="mm-key",
            MINIMAX_MODEL="MiniMax-M2.7-highspeed",
            MINIMAX_BASE_URL="https://api.minimaxi.com/v1",
        )
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, api_type = llm_client.read_api_config(
                None, None, None, None,
            )
        self.assertEqual(api_key, "mm-key")
        self.assertEqual(model, "MiniMax-M2.7-highspeed")
        self.assertEqual(base_url, "https://api.minimaxi.com/v1")
        self.assertIsNone(api_type)

    def test_baibaiaigc_env_aliases_still_resolve(self) -> None:
        env = _clear_env(os.environ)
        env.update(
            BAIBAIAIGC_API_KEY="legacy-key",
            BAIBAIAIGC_MODEL="legacy-model",
            BAIBAIAIGC_BASE_URL="https://legacy/v1",
        )
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = llm_client.read_api_config(
                None, None, None, None,
            )
        self.assertEqual(api_key, "legacy-key")
        self.assertEqual(model, "legacy-model")
        self.assertEqual(base_url, "https://legacy/v1")

    def test_minimax_wins_over_baibaiaigc_and_openai(self) -> None:
        env = _clear_env(os.environ)
        env.update(
            MINIMAX_API_KEY="mm-key",
            MINIMAX_MODEL="mm-model",
            MINIMAX_BASE_URL="https://api.minimaxi.com/v1",
            BAIBAIAIGC_API_KEY="legacy-key",
            BAIBAIAIGC_MODEL="legacy-model",
            BAIBAIAIGC_BASE_URL="https://legacy/v1",
            OPENAI_API_KEY="oai-key",
            OPENAI_BASE_URL="https://openai/v1",
        )
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = llm_client.read_api_config(
                None, None, None, None,
            )
        self.assertEqual(api_key, "mm-key")
        self.assertEqual(model, "mm-model")
        self.assertEqual(base_url, "https://api.minimaxi.com/v1")

    def test_explicit_args_still_win_over_env(self) -> None:
        env = _clear_env(os.environ)
        env.update(
            MINIMAX_API_KEY="mm-key",
            MINIMAX_MODEL="mm-model",
            MINIMAX_BASE_URL="https://api.minimaxi.com/v1",
        )
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = llm_client.read_api_config(
                "explicit-key", "explicit-model", "https://explicit/v1", None,
            )
        self.assertEqual(api_key, "explicit-key")
        self.assertEqual(model, "explicit-model")
        self.assertEqual(base_url, "https://explicit/v1")

    def test_openai_envs_still_fill_in_when_only_set(self) -> None:
        env = _clear_env(os.environ)
        env.update(
            OPENAI_API_KEY="oai-key",
            OPENAI_BASE_URL="https://openai/v1",
        )
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = llm_client.read_api_config(
                None, None, None, None,
            )
        self.assertEqual(api_key, "oai-key")
        self.assertIsNone(model)
        self.assertEqual(base_url, "https://openai/v1")


class ResolveApiConfigEnvFirstTests(unittest.TestCase):
    """:func:`deai_api._resolve_api_config` prefers env when fully set."""

    def setUp(self) -> None:
        # Always pretend the app config file does not exist so the
        # ~/.baibaiaigc fallback never fires during these tests.
        self.config_patch = patch.object(
            deai_api,
            "get_app_config_path",
            return_value=Path("/dev/null/no-such-file"),
        )
        self.config_patch.start()

    def tearDown(self) -> None:
        self.config_patch.stop()

    def test_env_wins_over_forwarded_payload_when_env_is_full(self) -> None:
        """If the container has its own provider configured, forwarded creds are ignored."""
        env = _clear_env(os.environ)
        env.update(
            MINIMAX_API_KEY="mm-key",
            MINIMAX_MODEL="MiniMax-M2.7-highspeed",
            MINIMAX_BASE_URL="https://api.minimaxi.com/v1",
        )
        payload = {
            "api_key": "forwarded-ark-key",
            "model": "deepseek-r1-250528",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        }
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = deai_api._resolve_api_config(payload)
        self.assertEqual(api_key, "mm-key")
        self.assertEqual(model, "MiniMax-M2.7-highspeed")
        self.assertEqual(base_url, "https://api.minimaxi.com/v1")

    def test_payload_wins_when_env_is_empty(self) -> None:
        """Legacy behaviour: forwarded creds drive the service when env is empty."""
        env = _clear_env(os.environ)
        payload = {
            "api_key": "forwarded-key",
            "model": "forwarded-model",
            "base_url": "https://forwarded/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = deai_api._resolve_api_config(payload)
        self.assertEqual(api_key, "forwarded-key")
        self.assertEqual(model, "forwarded-model")
        self.assertEqual(base_url, "https://forwarded/v1")

    def test_partial_env_falls_through_to_payload_per_field(self) -> None:
        """A container with only api_key set still merges with payload model / base_url."""
        env = _clear_env(os.environ)
        env.update(MINIMAX_API_KEY="mm-key")
        payload = {
            "model": "forwarded-model",
            "base_url": "https://forwarded/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = deai_api._resolve_api_config(payload)
        self.assertEqual(api_key, "mm-key")
        self.assertEqual(model, "forwarded-model")
        self.assertEqual(base_url, "https://forwarded/v1")

    def test_baibaiaigc_env_also_overrides_payload_when_full(self) -> None:
        env = _clear_env(os.environ)
        env.update(
            BAIBAIAIGC_API_KEY="legacy-key",
            BAIBAIAIGC_MODEL="legacy-model",
            BAIBAIAIGC_BASE_URL="https://legacy/v1",
        )
        payload = {
            "api_key": "forwarded",
            "model": "forwarded",
            "base_url": "https://forwarded/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = deai_api._resolve_api_config(payload)
        self.assertEqual(api_key, "legacy-key")
        self.assertEqual(model, "legacy-model")
        self.assertEqual(base_url, "https://legacy/v1")

    def test_camelcase_payload_fields_recognized(self) -> None:
        env = _clear_env(os.environ)
        payload = {
            "apiKey": "camel-key",
            "model": "camel-model",
            "baseUrl": "https://camel/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            api_key, model, base_url, _ = deai_api._resolve_api_config(payload)
        self.assertEqual(api_key, "camel-key")
        self.assertEqual(model, "camel-model")
        self.assertEqual(base_url, "https://camel/v1")


if __name__ == "__main__":
    unittest.main()
