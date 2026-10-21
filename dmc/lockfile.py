"""Execution lockfile: everything that can change an output's bits, pinned and hashed.

A compiler is trusted because the same source + the same toolchain gives the same binary.
The lockfile is our "toolchain version": two runs may be compared for bitwise parity only
if their lock hashes are equal. Any field change is a new toolchain and must be re-certified.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .canonical import content_hash, file_hash

# Environment variables that are known to change numerics or kernel selection.
DETERMINISM_ENV_VARS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "VLLM_BATCH_INVARIANT",
    "VLLM_ATTENTION_BACKEND",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
)


class ModelLock(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    revision: str | None = None
    weights_hash: str
    tokenizer_hash: str
    chat_template_hash: str | None = None
    dtype: str
    adapter_hash: str | None = None  # LoRA / task-profile adapter, if any


class RuntimeLock(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str  # e.g. "transformers", "vllm", "sglang"
    engine_version: str
    torch_version: str | None = None
    device: str  # e.g. "cpu", "cuda:H100-SXM-80GB"
    world: dict[str, int] = Field(default_factory=dict)  # tp / ep / pp / dp sizes
    flags: dict[str, str] = Field(default_factory=dict)  # determinism-relevant flags
    env: dict[str, str] = Field(default_factory=dict)
    python: str = Field(default_factory=platform.python_version)


class Lockfile(BaseModel):
    """The full toolchain identity of one execution profile."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "dmc.lock/1"
    model: ModelLock
    runtime: RuntimeLock
    ir_schema_version: str
    grammar_hash: str | None = None
    decoding: dict[str, object] = Field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self)


def capture_env() -> dict[str, str]:
    return {k: os.environ[k] for k in DETERMINISM_ENV_VARS if k in os.environ}


def fingerprint_hf_snapshot(path: str | Path) -> dict[str, str | None]:
    """Hash weights, tokenizer and chat template files of a local HF snapshot."""
    p = Path(path)
    weights = sorted(p.glob("*.safetensors")) or sorted(p.glob("*.bin"))
    if not weights:
        raise FileNotFoundError(f"no weight files under {p}")
    weights_hash = content_hash([file_hash(str(w)) for w in weights])
    tok_files = [p / n for n in ("tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt")]
    tokenizer_hash = content_hash([file_hash(str(t)) for t in tok_files if t.exists()])
    template = p / "tokenizer_config.json"
    chat_template_hash = file_hash(str(template)) if template.exists() else None
    return {
        "weights_hash": weights_hash,
        "tokenizer_hash": tokenizer_hash,
        "chat_template_hash": chat_template_hash,
    }
