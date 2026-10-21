"""Content-addressed result store (contract level D0: replay).

Keys are ``H(lock hash, IR hash, profile version, seed)``. A stored result is returned
byte-for-byte on replay; a stored result whose recorded lock differs from the caller's is
never returned, because it was produced by a different toolchain.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .canonical import canonical_json, content_hash


class Receipt(BaseModel):
    """Provenance attached to every result: what produced it and how confident it was."""

    model_config = ConfigDict(frozen=True)

    key: str
    lock_hash: str
    ir_hash: str  # covers the IR context, including session_state_hash
    session_state_hash: str | None = None
    rendered_prompt_hash: str | None = None  # bytes actually fed to the model
    profile: str
    profile_version: str
    seed: int | None
    margins: dict[str, float] = {}
    coverage_band: float | None = None  # abstention threshold in force
    verifier: dict[str, str] = {}
    fragile_routing_share: float | None = None
    routing_trace_ref: str | None = None  # content address of the routing trace (R27)
    pinned: str | None = None  # "full" | "partial" | "none" for re-executions
    sampling_mask: str | None = None  # summary of the truncation support used when sampling
    normalization: str | None = None  # Judge style-normalization mode
    cap_hits: int = 0  # tool outputs truncated to their compiled cap (dmc.scheduler.cap_input)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True)

    receipt: Receipt
    output: str  # canonical JSON of the profile's typed result


def result_key(lock_hash: str, ir_hash: str, profile_version: str, seed: int | None) -> str:
    return content_hash(
        {"lock": lock_hash, "ir": ir_hash, "profile": profile_version, "seed": seed}
    )


class ResultStore:
    """A directory of immutable records named by their key."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = key.split(":", 1)[1]
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str, *, lock_hash: str) -> Record | None:
        path = self._path(key)
        if not path.exists():
            return None
        record = Record.model_validate_json(path.read_text(encoding="utf-8"))
        if record.receipt.lock_hash != lock_hash:
            return None
        return record

    def put(self, record: Record) -> None:
        path = self._path(record.receipt.key)
        if path.exists():
            existing = Record.model_validate_json(path.read_text(encoding="utf-8"))
            if existing.output != record.output:
                # Same key, different bytes: the toolchain is not deterministic. Never overwrite.
                raise NondeterminismError(record.receipt.key, existing.output, record.output)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_json(json.loads(record.model_dump_json()))
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)


class NondeterminismError(RuntimeError):
    """Raised when a recomputation under an identical key produced different output."""

    def __init__(self, key: str, stored: str, recomputed: str) -> None:
        super().__init__(f"recomputation under {key} diverged from the stored result")
        self.key = key
        self.stored = stored
        self.recomputed = recomputed
