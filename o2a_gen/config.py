"""Generator configuration (loaded from a YAML file, see gen_config.example.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

MODEL_ROLES = (
    "extract",          # procedure document -> structured steps
    "ground",           # step + retrieved catalog docs -> agent fields (SQL, transform, prompt)
    "navigate",         # browses the compiled catalog skill tree
    "catalog_summary",  # catalog compile: folder names and summaries
    "catalog_cards",    # catalog compile: per-document title, summary, keywords
    "llm_agent",        # value written into `model:` of generated LlmAgent YAMLs
)

DEFAULT_EMBEDDING = {"provider": "local", "model": "Qwen/Qwen3-Embedding-0.6B"}


@dataclass
class GenConfig:
    llm: dict[str, Any]
    models: dict[str, str]
    embedding: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_EMBEDDING))
    catalog: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)

    def model(self, role: str) -> str:
        try:
            return self.models[role]
        except KeyError:
            raise KeyError(f"models.{role} is not set in the config") from None

    # generation.* accessors with defaults
    @property
    def prefix(self) -> str:
        return str(self.generation.get("prefix", "")).strip("_")

    @property
    def pipeline_inputs(self) -> list[str]:
        return list(self.generation.get("pipeline_inputs", []))

    @property
    def default_db_type(self) -> str:
        return str(self.generation.get("default_db_type", "teradata"))

    @property
    def default_connection_env(self) -> str | None:
        return self.generation.get("default_connection_env")

    @property
    def navigate_max_turns(self) -> int:
        return int(self.generation.get("navigate_max_turns", 12))

    @property
    def agent_templates(self) -> dict[str, dict]:
        return dict(self.generation.get("agent_templates", {}) or {})


def load_config(path: str | Path) -> GenConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    missing = [r for r in MODEL_ROLES if r not in (raw.get("models") or {})]
    if missing:
        raise ValueError(f"{path}: models is missing roles {missing}")
    return GenConfig(
        llm=raw.get("llm") or {},
        models=raw["models"],
        embedding=raw.get("embedding") or dict(DEFAULT_EMBEDDING),
        catalog=raw.get("catalog") or {},
        generation=raw.get("generation") or {},
    )
