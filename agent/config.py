"""Central configuration for SurgicalAgent.

The checked-in JSON file contains no secrets.  API credentials are resolved
only from the environment variable named by ``llm.api_key_env``.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是 JSON 对象：{path}")
    return value


class AppConfig:
    """Small dependency-free configuration wrapper."""

    def __init__(self, values: Dict[str, Any], source_path: Path):
        self.values = values
        self.source_path = source_path

    def section(self, name: str) -> Dict[str, Any]:
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"配置项 {name} 必须是对象")
        return value

    def path(self, name: str) -> Path:
        value = self.section("paths").get(name)
        if not value:
            raise KeyError(f"缺少路径配置：paths.{name}")
        path = Path(os.path.expandvars(os.path.expanduser(str(value))))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def llm_api_key(self) -> str | None:
        variable = str(self.section("llm").get("api_key_env", "")).strip()
        return os.getenv(variable) if variable else None

    def public_summary(self) -> Dict[str, Any]:
        """Return configuration metadata without ever including a secret."""
        values = deepcopy(self.values)
        values.get("llm", {}).pop("api_key", None)
        values.setdefault("runtime", {})["api_key_configured"] = bool(
            self.llm_api_key()
        )
        values["runtime"]["config_source"] = str(self.source_path)
        return values


_CONFIG: AppConfig | None = None


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    default_values = _read_json(DEFAULT_CONFIG_PATH)
    configured_path = path or os.getenv("SURGICAL_AGENT_CONFIG")
    source_path = DEFAULT_CONFIG_PATH
    values = default_values
    if configured_path:
        source_path = Path(configured_path).expanduser().resolve()
        values = _deep_merge(default_values, _read_json(source_path))

    # Environment overrides are intentionally limited and never include the
    # API key value itself.
    overrides = {
        "SURGICAL_MODEL_PATH": ("paths", "model_checkpoint", str),
        "SURGICAL_DEVICE": ("vision", "device", str),
        "SURGICAL_THRESHOLD": ("vision", "component_threshold", float),
        "SURGICAL_TRIPLET_THRESHOLD": ("vision", "triplet_threshold", float),
        "SURGICAL_SAMPLE_INTERVAL": ("video", "sample_interval", float),
        "SURGICAL_LLM_BASE_URL": ("llm", "base_url", str),
        "SURGICAL_LLM_MODEL": ("llm", "model", str),
    }
    for env_name, (section, key, cast) in overrides.items():
        raw = os.getenv(env_name)
        if raw is not None:
            values.setdefault(section, {})[key] = cast(raw)

    # Backward compatibility with the original project variable.
    if os.getenv("DEEPSEEK_MODEL") and not os.getenv("SURGICAL_LLM_MODEL"):
        values.setdefault("llm", {})["model"] = os.environ["DEEPSEEK_MODEL"]
    return AppConfig(values, source_path)


def get_config(reload: bool = False) -> AppConfig:
    global _CONFIG
    if _CONFIG is None or reload:
        _CONFIG = load_config()
    return _CONFIG

