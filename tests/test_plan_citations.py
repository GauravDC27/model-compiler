"""Every code, test, experiment and result path (and `path::symbol`) cited in the plan and its renderings
must exist (reviews G3, round-7 Q5)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITE = re.compile(r"`?((?:dmc|gcam|tests|experiments)/[\w/.-]+?\.(?:py|md|json))(?:::(\w+))?`?")
DOCS = ("docs/PLAN.md", "docs/blueprint.html", "docs/pdf/build_blueprint_pdf.py", "docs/GCAM_V1.md",
        "docs/GCAM_BENCHMARK.md")


def test_cited_files_and_symbols_exist():
    text = "\n".join((ROOT / d).read_text(encoding="utf-8") for d in DOCS)
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
    assert not missing, f"the docs cite things that do not exist: {sorted(missing)}"


def test_cited_receipt_and_context_fields_exist():
    """Attribute claims like `Receipt.rendered_prompt_hash` must name real fields (review K15)."""
    from dmc.ir import Context
    from dmc.store import Receipt

    text = (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    models = {"Receipt": Receipt, "Context": Context}
    missing = [f"{cls}.{attr}" for cls, attr in re.findall(r"\b(Receipt|Context)\.(\w+)", text)
               if attr not in models[cls].model_fields]
    assert not missing, f"PLAN.md cites fields that do not exist: {sorted(set(missing))}"


def test_ir_op_count_matches_the_plan():
    from dmc.ir import op_json_schema

    defs = op_json_schema()["$defs"].values()
    ops = {d["properties"]["op"]["const"] for d in defs if "op" in d.get("properties", {})}
    assert "All twelve ops exist as types" in (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    assert len(ops) == 12
