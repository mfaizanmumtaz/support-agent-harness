"""The sample business used by the demo, the evals and the tests: Northwind Hotel (fictional)."""

import json
from pathlib import Path

from harness.contracts import AgentSpec

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "northwind_hotel"
CACHE = ROOT / ".cache"


def load_agent(folder: Path = SAMPLE) -> AgentSpec:
    return AgentSpec(**json.loads((folder / "agent.json").read_text(encoding="utf-8")))


def knowledge_folder(folder: Path = SAMPLE) -> Path:
    return folder / "knowledge"
