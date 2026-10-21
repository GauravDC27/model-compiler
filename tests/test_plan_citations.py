"""Every `dmc/...py` path and `dmc/...py::symbol` cited in docs/PLAN.md must exist (review G3)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITE = re.compile(r"`?((?:dmc|tests|experiments)/[\w/]+\.(?:py|md|json))(?:::(\w+))?`?")


def test_cited_files_and_symbols_exist():
    text = (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    missing = []
    for path, symbol in set(CITE.findall(text)):
        f = ROOT / path
        if not f.exists():
            missing.append(path)
            continue
        if symbol:
            src = f.read_text(encoding="utf-8")
            if not re.search(rf"^\s*(?:def|class)\s+{symbol}\b|^{symbol}\s*[:=]", src, re.M):
                missing.append(f"{path}::{symbol}")
    assert not missing, f"PLAN.md cites things that do not exist: {sorted(missing)}"
