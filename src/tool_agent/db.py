"""SQLite storage for the model catalog: schema, writes for the seeder, reads for the tools."""

import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from tool_agent.schemas import ModelInfo

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "pricing.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id              TEXT PRIMARY KEY,
    provider              TEXT NOT NULL,
    name                  TEXT NOT NULL,
    description           TEXT NOT NULL,
    context_length        INTEGER NOT NULL,
    max_output_tokens     INTEGER,
    input_modalities      TEXT NOT NULL,  -- comma-separated, e.g. "text,image"
    supports_tools        INTEGER NOT NULL,
    supports_reasoning    INTEGER NOT NULL,
    intelligence_index    REAL,
    input_per_mtok        REAL NOT NULL,
    output_per_mtok       REAL NOT NULL,
    cached_input_per_mtok REAL
);
"""

COLUMNS = list(ModelInfo.model_fields)


def db_path() -> Path:
    """TOOL_AGENT_DB overrides the default, which is how the tests point at a temp file."""
    return Path(os.environ.get("TOOL_AGENT_DB", DEFAULT_DB_PATH))


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or db_path()
    if not path.exists():
        raise FileNotFoundError(f"No model database at {path}. Run: uv run tool-agent-setup")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def create_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)


def replace_models(path: Path, models: Iterable[ModelInfo]) -> int:
    """Replace the whole catalog in one transaction, so a failed seed leaves the old one."""
    rows = [_to_row(m) for m in models]
    placeholders = ", ".join("?" for _ in COLUMNS)
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM models")
        conn.executemany(f"INSERT INTO models ({', '.join(COLUMNS)}) VALUES ({placeholders})", rows)
    return len(rows)


def get_model(model_id: str) -> ModelInfo | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
    return _from_row(row) if row else None


def all_model_ids() -> list[str]:
    with connect() as conn:
        return [r[0] for r in conn.execute("SELECT model_id FROM models ORDER BY model_id")]


def find_models(
    *,
    provider: str | None,
    min_context: int | None,
    needs_tools: bool,
    needs_reasoning: bool,
    needs_image_input: bool,
    max_input_per_mtok: float | None,
    order_by: str,
) -> list[ModelInfo]:
    where: list[str] = []
    params: list[object] = []
    if provider:
        where.append("provider = ?")
        params.append(provider)
    if min_context:
        where.append("context_length >= ?")
        params.append(min_context)
    if needs_tools:
        where.append("supports_tools = 1")
    if needs_reasoning:
        where.append("supports_reasoning = 1")
    if needs_image_input:
        where.append("(',' || input_modalities || ',') LIKE '%,image,%'")
    if max_input_per_mtok is not None:
        where.append("input_per_mtok <= ?")
        params.append(max_input_per_mtok)
    sql = "SELECT * FROM models"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_by}"  # order_by comes from a fixed map in tools.py, never the model
    with connect() as conn:
        return [_from_row(r) for r in conn.execute(sql, params)]


def _to_row(m: ModelInfo) -> tuple[object, ...]:
    data = m.model_dump()
    data["input_modalities"] = ",".join(m.input_modalities)
    return tuple(data[c] for c in COLUMNS)


def _from_row(row: sqlite3.Row) -> ModelInfo:
    data = dict(row)
    data["input_modalities"] = data["input_modalities"].split(",")
    return ModelInfo.model_validate(data)
