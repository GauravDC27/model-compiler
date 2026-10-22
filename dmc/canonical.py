"""Canonical serialization and content addressing.

Every artifact that can influence an output (IR, lockfile, decoding policy, result) is
serialized canonically before hashing, so equal values always produce equal keys.
The JSON form follows the RFC 8785 (JCS) rules that matter for our value space:
sorted keys, no insignificant whitespace, UTF-8, and a single canonical float form.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from pydantic import BaseModel


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not canonicalizable: {value!r}")
        # Integral floats collapse to ints, as in JCS / ECMAScript number serialization.
        if value.is_integer() and abs(value) < 2**53:
            return int(value)
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"type {type(value).__name__} is not canonicalizable")


def canonical_json(value: Any) -> str:
    """Return the canonical JSON text for ``value``."""
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any, *, prefix: str = "sha256") -> str:
    """Return a stable content address for ``value`` (``sha256:<hex>``)."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def file_hash(path: str, *, chunk: int = 1 << 20) -> str:
    """Hash a file's bytes (used for weights, tokenizer and template fingerprints)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return f"sha256:{h.hexdigest()}"
