"""Build docs/GCAM_Blueprint.pdf from docs/blueprint.html.

The sheets (and their inline SVG drawings) are taken verbatim from the web blueprint, so the two
never drift apart. The PDF adds a cover with a title block and revision history, a contents page
with the diagram legend, an executive summary, specification tables, and appendices (risk
register from PLAN.md §I.7, glossary, review trail and evidence base).

Rendering: headless Chromium through the globally installed Playwright (Node), A4 landscape,
vector SVG, fonts embedded from local files. Two passes: the first finds the page each section
starts on, the second prints those page numbers into the contents.

    python3 docs/pdf/build_blueprint_pdf.py <font_dir>

``font_dir`` holds the Google Fonts files and a ``local.css`` whose @font-face rules use
relative file names (see ``fetch_fonts``).
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/blueprint.html"
PLAN = ROOT / "docs/PLAN.md"
OUT = ROOT / "docs/GCAM_Blueprint.pdf"
FONTS_CSS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800"
             "&family=IBM+Plex+Mono:wght@400;500;600"
             "&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"

PLAN_VERSION = "v6.2"
DATE = "2026-10-21"
REVIEW_STATUS = "Round 6: approved with changes, no blockers (all applied). Round 7 in progress."


def fetch_fonts(font_dir: Path) -> None:
    """Download the latin, latin-ext and greek subsets and write local.css."""
    font_dir.mkdir(parents=True, exist_ok=True)
    css = subprocess.run(["curl", "-sS", "-A", UA, FONTS_CSS], check=True, capture_output=True, text=True).stdout
    keep = []
    for subset, block in re.findall(r"/\* ([a-z-]+) \*/\s*(@font-face \{.*?\})", css, re.S):
        if subset not in ("latin", "latin-ext", "greek"):
            continue
        url, fmt = re.search(r"url\((https://[^)]+)\) format\('([a-z0-9]+)'\)", block).groups()
        name = "f_" + hashlib.sha1(url.encode()).hexdigest()[:12] + (".woff2" if fmt == "woff2" else ".ttf")
        if not (font_dir / name).exists():
            subprocess.run(["curl", "-sS", "-o", str(font_dir / name), url], check=True)
        keep.append(block.replace(url, name))
    (font_dir / "local.css").write_text("\n".join(keep))


def md_inline(text: str) -> str:
    """The little Markdown the plan's tables use: bold and code spans."""
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


def risk_rows() -> list[list[str]]:
    plan = PLAN.read_text(encoding="utf-8")
    block = plan[plan.index("## I.7 Risks"):plan.index("# Part II")]
    rows = [r.strip().strip("|").split("|") for r in block.splitlines() if r.startswith("| ")]
    return [[c.strip() for c in r] for r in rows[1:] if not set("".join(r)) <= set("-: ")]


PRINT_CSS = """
@page { size: A4 landscape; margin: 12mm 14mm 14mm 14mm; }
html, body { background: #FFFFFF !important; }
body { background-image: none !important; font-size: 10.4pt; line-height: 1.45;
       -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.marker { color: #FFFFFF; font-size: 1px; line-height: 1px; position: absolute; left: 0; top: 0; }
.page { break-before: page; position: relative; }
.sheet { margin-top: 0; break-before: page; position: relative; }
.sheet-h { padding-bottom: 5px; }
.sheet-h h2 { font-size: 21pt; }
.claim { font-size: 11.2pt; max-width: none; margin-top: 7px; }
figure { margin-top: 10px; break-inside: avoid; border: 0.75pt solid var(--rule); }
.fig { overflow: visible; padding: 6px 8px; }
.fig svg, .fig.narrow svg { min-width: 0 !important; width: 100%; max-width: 100%; max-height: 122mm; margin: 0 auto; }
.fig.narrow svg { max-height: 118mm; }
#s2 .fig svg, #s10 .fig svg { max-height: 64mm; }
#s6 .fig svg { max-height: 84mm; }
figcaption { font-size: 8.8pt; padding: 6px 12px 8px; }
.cols { grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px 26px; font-size: 9.4pt; margin-top: 12px; break-inside: avoid; }
.cols h3 { font-size: 11pt; }
.tbl { overflow: visible; margin-top: 12px; }
table { min-width: 0; font-size: 8.4pt; line-height: 1.36; }
th, td { padding: 3.4px 8px; }
th { font-size: 7.6pt; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
.num { font-size: 8.4pt; }
h3.sub { font-family: var(--display); font-stretch: 88%; font-size: 11.5pt; margin-top: 14px; }

/* Cover */
.cover { height: 182mm; display: grid; grid-template-rows: auto 1fr auto; border: 1.6pt solid var(--ink); overflow: hidden; }
.cover-top { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); border-bottom: 1.6pt solid var(--ink); }
.cover-main { padding: 8mm 12mm 6mm; border-right: 1.6pt solid var(--ink); display: flex; flex-direction: column; gap: 4mm; }
.cover-main h1 { font-size: 38pt; line-height: .98; }
.cover-main .sub { font-family: var(--display); font-stretch: 88%; font-size: 13pt; color: var(--muted); max-width: 60ch; }
.cover-main p { max-width: 78ch; font-size: 9.6pt; }
.cover .tb-cells { font-size: 9pt; align-content: start; }
.cover .tb-cells div { padding: 3.6px 10px; }
.cover-mid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); }
.cover-index { padding: 4mm 12mm; border-right: 1.6pt solid var(--ink); }
.cover-rev { padding: 4mm 8mm; }
.cover h2.blk { font-family: var(--mono); font-size: 8pt; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 500; margin-bottom: 3mm; }
.idx { columns: 2; column-gap: 9mm; font-family: var(--display); font-stretch: 86%; font-size: 9.4pt; list-style: none; padding: 0; margin: 0; }
.idx li { display: flex; gap: 8px; padding: 1.6px 0; border-bottom: 0.5pt dashed var(--rule); break-inside: avoid; }
.idx .n { font-family: var(--mono); font-size: 8.4pt; color: var(--muted); min-width: 2.2em; padding-top: 1px; }
.idx .t { flex: 1; }
.idx .p { font-family: var(--mono); font-size: 8.4pt; color: var(--muted); }
.rev { width: 100%; font-size: 7.8pt; min-width: 0; line-height: 1.3; }
.rev td, .rev th { padding: 2px 5px; }
.cover-foot { border-top: 1.6pt solid var(--ink); display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }
.cover-foot div { padding: 2.6mm 5mm; border-right: 1pt solid var(--rule); }
.cover-foot div:last-child { border-right: none; }
.cover-foot dt { font-family: var(--mono); font-size: 7.4pt; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.cover-foot dd { margin: 2px 0 0; font-family: var(--display); font-stretch: 85%; font-weight: 650; font-size: 10.6pt; line-height: 1.2; }

/* Contents + legend */
.two { margin-top: 4mm; }
.toc { columns: 2; column-gap: 12mm; }
.toc li { break-inside: avoid; }
.toc { list-style: none; padding: 0; margin: 0; font-family: var(--display); font-stretch: 88%; font-size: 10.4pt; }
.toc li { display: flex; gap: 10px; align-items: baseline; padding: 3px 0; border-bottom: 0.5pt dashed var(--rule); }
.toc .n { font-family: var(--mono); font-size: 8.8pt; color: var(--accent); min-width: 2.6em; }
.toc .t { flex: 1; }
.toc .d { display: block; font-family: var(--body); font-stretch: 100%; font-size: 8pt; color: var(--muted); line-height: 1.3; }
.toc .p { font-family: var(--mono); font-size: 9pt; }
.legend { display: grid; grid-template-columns: repeat(3, 56px minmax(0, 1fr)); gap: 6px 10px; align-items: center; font-size: 8.4pt; line-height: 1.3; }
.legend svg { width: 56px; height: 23px; }
.note { font-size: 8.8pt; color: var(--muted); margin-top: 3mm; }

/* Executive summary */
.exec-grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 9mm; margin-top: 4mm; font-size: 9.6pt; line-height: 1.42; }
.exec-grid p + p { margin-top: 6px; }
.exec-grid ul { margin: 4px 0 0; padding-left: 1.1em; }
.exec-grid li + li { margin-top: 3px; }
.exec-grid h3 { font-family: var(--display); font-stretch: 88%; font-size: 11.4pt; margin: 8px 0 3px; }
.exec-grid h3:first-child { margin-top: 0; }
.kv { width: 100%; min-width: 0; font-size: 8.9pt; }
.state { font-family: var(--mono); font-size: 7.6pt; letter-spacing: .04em; padding: 0 6px; border-radius: 8px; border: 0.6pt solid; white-space: nowrap; }
.state.d { color: var(--ok); background: var(--ok-soft); }
.state.o { color: var(--warn); background: var(--warn-soft); }
.gloss { columns: 3; column-gap: 9mm; font-size: 8.8pt; line-height: 1.38; margin-top: 4mm; }
.gloss dt { font-family: var(--display); font-stretch: 88%; font-weight: 700; font-size: 9.8pt; break-after: avoid; }
.gloss dd { margin: 0 0 6px; break-inside: avoid; }
"""


def legend_svg(kind: str) -> str:
    shapes = {
        "box": '<rect x="2" y="3" width="60" height="20" rx="3" fill="var(--panel)" stroke="var(--ink)" stroke-width="1.2"/>',
        "hi": '<rect x="2" y="3" width="60" height="20" rx="3" fill="var(--accent-soft)" stroke="var(--accent)" stroke-width="1.6"/>',
        "frame": '<rect x="2" y="3" width="60" height="20" rx="3" fill="none" stroke="var(--accent)" stroke-width="1.5" stroke-dasharray="7 4"/>',
        "dash": '<rect x="2" y="3" width="60" height="20" rx="3" fill="var(--panel)" stroke="var(--muted)" stroke-width="1.2" stroke-dasharray="5 4"/>',
        "arrow": '<line x1="4" y1="13" x2="54" y2="13" stroke="var(--ink)" stroke-width="1.3"/><path d="M52,8 L62,13 L52,18 z" fill="var(--ink)"/>',
        "acc": '<line x1="4" y1="13" x2="54" y2="13" stroke="var(--accent)" stroke-width="1.7"/><path d="M52,8 L62,13 L52,18 z" fill="var(--accent)"/>',
        "warn": '<line x1="4" y1="13" x2="54" y2="13" stroke="var(--warn)" stroke-width="1.7"/><path d="M52,8 L62,13 L52,18 z" fill="var(--warn)"/>',
        "ok": '<rect x="2" y="5" width="18" height="16" rx="2" fill="var(--ok-soft)" stroke="var(--ok)"/><rect x="23" y="5" width="18" height="16" rx="2" fill="var(--warn-soft)" stroke="var(--warn)"/><rect x="44" y="5" width="18" height="16" rx="2" fill="var(--bad-soft)" stroke="var(--bad)"/>',
        "diamond": '<path d="M32,4 L41,13 L32,22 L23,13 Z" fill="var(--warn-soft)" stroke="var(--warn)" stroke-width="1.2"/>',
    }
    return f'<svg viewBox="0 0 64 26" aria-hidden="true">{shapes[kind]}</svg>'


def build_html(sheets: list[tuple[str, str, str]], style: str, font_css: str, pages: dict[str, int]) -> str:
    def pg(key: str) -> str:
        return str(pages.get(key, "")) if pages else ""

    sheet_titles = [(sid, title) for sid, title, _ in sheets]
    idx_items = "".join(
        f'<li><span class="n">{sid[1:].zfill(2)}</span><span class="t">{html.escape(t)}</span><span class="p">{pg(sid)}</span></li>'
        for sid, t in sheet_titles)
    sheet_desc = {
        "s1": "Compiler, runtime, executor tiers and trust layer, and the arrows between them.",
        "s2": "An example goal laid out on two lanes with worst-case bars, a guard and a variant.",
        "s3": "Why ladders are priced item by item: the scheduler's own correlated-errors test.",
        "s4": "What happens when a guard fails, and how the program bound adds up.",
        "s5": "Router, goal bias, base, hot and goal experts in one MoE layer of the hub.",
        "s6": "Per-family checkpoints versus one goal-gated checkpoint, and the rule that picks.",
        "s7": "T0–T6, the agentic spike, and the edges that keep atlases and heads fresh.",
        "s8": "Cost-of-pass as a function of the escalation budget, with the targets drawn in.",
        "s9": "Replay versus recompute, the determinism levels, and each executor's contract.",
        "s10": "G0–G4 with their gates, the two allocation tiers, and the G0 measurements.",
        "s11": "Modules, tests and the status of every CPU pilot.",
    }
    toc = (
        f'<li><span class="n">—</span><span class="t">Executive summary<span class="d">Goal, approach, targets, decisions and open questions.</span></span><span class="p">{pg("exec")}</span></li>'
        + "".join(
            f'<li><span class="n">{sid[1:].zfill(2)}</span><span class="t">{html.escape(t)}<span class="d">{sheet_desc[sid]}</span></span><span class="p">{pg(sid)}</span></li>'
            for sid, t in sheet_titles)
        + f'<li><span class="n">A</span><span class="t">Risk register<span class="d">Every Part I risk with likelihood and mitigation (PLAN.md §I.7).</span></span><span class="p">{pg("appA")}</span></li>'
        + f'<li><span class="n">B</span><span class="t">Glossary<span class="d">Terms, symbols and stage names used on the sheets.</span></span><span class="p">{pg("appB")}</span></li>'
        + f'<li><span class="n">C</span><span class="t">Review trail and evidence base<span class="d">Six adversarial review rounds and the sources behind every number.</span></span><span class="p">{pg("appC")}</span></li>'
    )

    cover = f"""
<section class="cover"><span class="marker">@@cover@@</span>
  <div class="cover-top">
    <div class="cover-main">
      <span class="eyebrow">Implementation blueprint · drawing set</span>
      <h1>Goal-Compiled <em>Agentic MoE</em></h1>
      <span class="sub">A goal compiler, a goal-conditioned MoE hub and a compiler-grade trust layer, specified in eleven sheets.</span>
      <p>A goal compiler turns each request into a typed task graph and a static schedule. Every step's executor, token budget, escalation ladder, guards and worst-case cost are decided before anything runs. An assembly of executors runs that schedule: compiled code, System-1 models, classic ML, a Qwen3.6-35B-A3B hub with goal-family capacity, and a Qwen3.8-27B spoke. Everything runs under a determinism contract with replay, receipts and declared guarantees per executor.</p>
    </div>
    <div class="tb-cells">
      <div>Project</div><div>GCAM · Goal-Compiled Agentic MoE</div>
      <div>Plan</div><div>{PLAN_VERSION} · <span class="mono">docs/PLAN.md</span></div>
      <div>Date</div><div class="mono">{DATE}</div>
      <div>Hub</div><div>Qwen3.6-35B-A3B (35B total, ~3B active)</div>
      <div>Reference</div><div>Qwen3.8-27B (dense)</div>
      <div>Review</div><div>{REVIEW_STATUS}</div>
      <div>Repository</div><div class="mono">gauravdc27/model-compiler</div>
      <div>Branch</div><div class="mono">claude/magical-allen-eickyt</div>
      <div>Format</div><div>A4 landscape · {{PAGES}} pages · vector drawings</div>
    </div>
  </div>
  <div class="cover-mid">
    <div class="cover-index"><h2 class="blk">Drawing index</h2><ol class="idx">{idx_items}</ol></div>
    <div class="cover-rev"><h2 class="blk">Revision history</h2>
      <table class="rev"><thead><tr><th>Rev</th><th>Change</th></tr></thead><tbody>
        <tr><td class="num">v1–v4</td><td>Determinism contract D0–D⊥, requirements R1–R28, trust architecture; converged over review rounds 1–3</td></tr>
        <tr><td class="num">v5</td><td>Rethink as a goal-compiled agentic MoE on Qwen3.6-35B-A3B; Groq-style compile-time scheduling</td></tr>
        <tr><td class="num">v6</td><td>Feasibility-derived targets, served artifact A/B, G0 gates, agentic spike, allocation tiers (round 4)</td></tr>
        <tr><td class="num">v6.1</td><td>Per-suite cost targets, every-variant bound, roadmap follows stage edges, goal-suite size (round 5)</td></tr>
        <tr><td class="num">v6.2</td><td>Spike gate power, δ_b cap and pooling, switch rule on the spike checkpoint, gated confirmations (round 6)</td></tr>
      </tbody></table>
    </div>
  </div>
  <dl class="cover-foot">
    <div><dt>Quality target</dt><dd>Non-inferior to Qwen3.8-27B</dd></div>
    <div><dt>Cost target</dt><dd>≤ 0.5× reasoning · ≤ 0.6× agentic</dd></div>
    <div><dt>Escalation budget</dt><dd>≤ 15% of the 27B's tokens</dd></div>
    <div><dt>Built so far</dt><dd>Compiler core · 83 tests</dd></div>
  </dl>
</section>"""

    legend_rows = [
        ("box", "Component or step. Labels name it; the second line says what it holds or does."),
        ("hi", "The element the sheet is about: a compile-time decision, a goal expert, a release."),
        ("frame", "Compile-time boundary: everything inside is decided before the request runs."),
        ("dash", "Conditional or optional: an escalation rung, a variant, the spike, a Tier 0 extension."),
        ("arrow", "Flow of data or control, labelled with what moves. Double-headed: node in, value out."),
        ("acc", "A compiler-set input: the program, the goal bias, a pinned replica."),
        ("warn", "Abstention or failure path: escalate, take a variant, recompile."),
        ("ok", "Outcome colours in grids and states: correct or pass, abstain or escalate, wrong or failed."),
        ("diamond", "Guard: a typed predicate checked over a node's output."),
    ]
    legend = "".join(f"<div>{legend_svg(k)}</div><div>{html.escape(t)}</div>" for k, t in legend_rows)
    contents = f"""
<section class="page"><span class="marker">@@contents@@</span>
  <div class="sheet-h"><span class="sheet-no">Contents</span><h2>How this set is organised</h2><span class="ref">Source of truth: docs/PLAN.md {PLAN_VERSION}</span></div>
  <div class="two">
    <ol class="toc">{toc}</ol>
    <div>
      <h3 class="sub">Reading the drawings</h3>
      <div class="legend" style="margin-top:2mm">{legend}</div>
      <p class="note">Every drawing is one claim, stated above it and in its caption. Section references (§I.2.1, R26) point into PLAN.md; file names point into the repository. Numbers marked illustrative are worked examples; every other number is from the plan, the code's tests, the research notes or the A3B training report.</p>
    </div>
  </div>
</section>"""

    exec_summary = """
<section class="page"><span class="marker">@@exec@@</span>
  <div class="sheet-h"><span class="sheet-no">Summary</span><h2>Executive summary</h2><span class="ref">Plan §I.0, §I.1</span></div>
  <div class="exec-grid">
    <div>
      <h3>What is being built</h3>
      <p>An agentic model system that matches Qwen3.8-27B on reasoning and agentic work at a fraction of its cost, and that people can trust the way they trust a compiler. The base is Qwen3.6-35B-A3B, a 256-expert MoE with about 3B active parameters. The gap to close is mainly agentic: on its model card the hub trails the 27B by 3 points on GPQA-Diamond but by 12 on SWE-bench Pro.</p>
      <h3>How</h3>
      <ul>
        <li><strong>Compile, then run.</strong> A goal compiler fixes each step's executor, budget, escalation ladder, guards and worst-case cost before execution. This is Groq's discipline applied to agents: no runtime arbitration, one plan as the source of truth.</li>
        <li><strong>Cheapest adequate executor.</strong> Code, System-1 heads, encoders and classic ML take whole steps off the LLM and abstain upward only when calibration says they are unsure. Ladders are priced on joint calibration data, because escalated items are the hard ones.</li>
        <li><strong>Goal-based MoE.</strong> The goal selects capacity at compile time: hot-expert adapters per goal family (item i) and goal-gated new experts reached through a compiler-set router bias (item ii). With the goal off, a layer is bit-identical to the base.</li>
        <li><strong>Capability transfer.</strong> On-policy distillation from Qwen3.8-27B over the shared 248,320-token vocabulary, agentic RL with routing replay, and decision-token DPO, on top of the user's proven A3B recipe.</li>
        <li><strong>Trust.</strong> Replay is exact by construction. Each executor declares the determinism contract it delivers and each node the one it requires. Every call leaves a receipt.</li>
      </ul>
      <h3>Built so far</h3>
      <p>The scheduler (joint calibration, caps, guards, variants, recompiles, capability ceilings, confirmation for irreversible actions), goal-gated experts, hot-expert LoRA and the trust layer exist as tested Python (112 tests). GCAM v1 runs nine goal families end to end on CPU: compiled ladders match or beat a dense specialist on every decision op at under 0.2% of its cost on classification, and the hub's goal-family layers lift it from 0.27 to 0.96 and 0.50 to 0.98 on their families. Four CPU pilots are done and three are queued.</p>
    </div>
    <div>
      <h3>Targets (fixed before measurement)</h3>
      <table class="kv"><tbody>
        <tr><td>Reasoning</td><td>Lower 95% CI of (GCAM − 27B) ≥ −δ_b on GPQA-D, HLE, AIME 24–26, LCB v6, MMLU-Pro (δ_b capped at 4; AIME and GPQA-D expected resolution-limited, so pooled into a composite)</td></tr>
        <tr><td>Agentic</td><td>Same rule, floor 2 points: SWE-bench Pro, Terminal-Bench 2.x, τ³-bench, BrowseComp, goal suite</td></tr>
        <tr><td>Cost</td><td>≤ 0.5× cost-of-pass (reasoning, goal suite), ≤ 0.6× (agentic), escalation e ≤ 0.15</td></tr>
        <tr><td>Latency</td><td>Mean ≤ 0.6×; p50 and p90 reported</td></tr>
        <tr><td>Predictability</td><td>Worst-case bound by construction; realized within ±15% over ≥ 200 tasks</td></tr>
      </tbody></table>
      <h3>Decided, and open</h3>
      <table class="kv"><tbody>
        <tr><td><span class="state d">decided</span></td><td>Hub base, compile-time scheduling, joint calibration, per-family checkpoints served from G1, cost formula</td></tr>
        <tr><td><span class="state o">open · G1</span></td><td>Does RL move agentic quality at 3B active? Agentic spike; switch to Qwen3.8-Flash-Next if not</td></tr>
        <tr><td><span class="state o">open · G0</span></td><td>Hub cost per token r; prefix caching and determinism on Gated DeltaNet; thinking control; MTP</td></tr>
        <tr><td><span class="state o">open · G2</span></td><td>One goal-gated checkpoint (B) instead of per-family checkpoints (A)</td></tr>
        <tr><td><span class="state o">open · T2</span></td><td>Do new goal experts beat matched LoRA? Pilot 06 on CPU first</td></tr>
      </tbody></table>
    </div>
  </div>
</section>"""

    risks = risk_rows()
    risk_html = "".join(f"<tr><td>{md_inline(r[0])}</td><td>{md_inline(r[1])}</td><td>{md_inline(r[2])}</td></tr>" for r in risks)
    app_a = f"""
<section class="page"><span class="marker">@@appA@@</span>
  <div class="sheet-h"><span class="sheet-no">Appendix A</span><h2>Risk register</h2><span class="ref">Plan §I.7 (Part I) · Part II §II.8 covers the trust track</span></div>
  <div class="tbl"><table><thead><tr><th style="width:30%">Risk</th><th style="width:12%">Likelihood</th><th>Mitigation</th></tr></thead><tbody>{risk_html}</tbody></table></div>
</section>"""

    glossary = [
        ("A3B report", "The user's training report for Qwen3.6-35B-A3B voice agents (hot-expert LoRA, top-20 KD, decision-token DPO, two-judge labels). Its methods are carried over."),
        ("Artifact A / B", "What is served. A: one merged checkpoint per goal family, pinned to its own GPU. B: one checkpoint whose goal-gated experts carry every family. A from G1; B only if it passes a comparison at G2."),
        ("avg@k", "Accuracy averaged over k samples per item; reduces sampling noise in paired comparisons."),
        ("b_goal", "Per-request router bias set by the compiler: opens a family's goal experts, −∞ for every other family."),
        ("Cost-of-pass", "Cost per solved task. At accuracy parity its ratio to the 27B is r·τ_h + e + s (s: System-1 and code overhead)."),
        ("D0 … D4, D⊥", "Determinism levels: replay, bitwise recompute, cross-stack decision parity, paraphrase invariance, spec conformance, defined failure."),
        ("δ_b", "Non-inferiority margin for benchmark b: max(1.5 points, one item, 2.8 × paired SE of hub-base − 27B), capped at 4 points; benchmarks above the cap are pooled."),
        ("e", "Escalation budget: tokens the 27B spends on escalated steps ÷ the 27B's tokens per task. Planned ≤ 0.15."),
        ("ESFT", "Expert-specialized fine-tuning: tuning only the 5–15% of experts a task uses."),
        ("G0 … G4", "Roadmap phases: baselines and de-risking, transfer and compiler v1, goal capacity, agentic RL, certification."),
        ("GDN", "Gated DeltaNet, the linear-attention layers in 30 of the hub's 40 layers. Their determinism and prefix caching are G0 gates."),
        ("Guard, variant, recompile", "A guard is a typed check on a step's output. On failure the runtime runs a pre-compiled variant once, or recompiles within an allowance, or abstains."),
        ("Hot experts", "The experts a goal family's traffic selects most, found by selection counts in a routing atlas."),
        ("hub-base", "Qwen3.6-35B-A3B after T1 capability transfer, merged and exported to FP8; every family's capacity is trained on it."),
        ("Joint calibration", "Running every executor on the same calibration items and pricing a ladder by replaying it item by item (Outcome records)."),
        ("Lock, receipt", "The lock pins every output-affecting artifact; its hash enters every result key. A receipt records how each result was produced."),
        ("MTP", "Multi-token prediction head used for speculative decoding; unmeasured on GDN, off in the bitwise lane."),
        ("OPD", "On-policy distillation: the student samples, the teacher scores every token."),
        ("r", "The hub's serving cost per token over the 27B's, measured on our node in G0. Priors 1/3 and 1/6."),
        ("R3", "Routing replay: rollout routes reused in the RL update so MoE training stays stable."),
        ("System-1 / System-2", "Fast executors that answer without generating tokens, versus the hub and spokes that reason in tokens."),
        ("T0 … T6", "Training stages: atlas, transfer, goal experts, agentic RL, decision DPO, efficiency, System-1 distillation."),
        ("τ_h", "Hub tokens per task ÷ the 27B's. Planned ≤ 1.05 on reasoning, 1.2 on agentic suites."),
    ]
    gloss = "".join(f"<dt>{html.escape(t)}</dt><dd>{html.escape(d)}</dd>" for t, d in glossary)
    app_b = f"""
<section class="page"><span class="marker">@@appB@@</span>
  <div class="sheet-h"><span class="sheet-no">Appendix B</span><h2>Glossary</h2><span class="ref">Symbols and names used on the sheets</span></div>
  <dl class="gloss">{gloss}</dl>
</section>"""

    app_c = """
<section class="page"><span class="marker">@@appC@@</span>
  <div class="sheet-h"><span class="sheet-no">Appendix C</span><h2>Review trail and evidence base</h2><span class="ref">docs/reviews/ · research_notes/</span></div>
  <p class="claim">Every plan version was reviewed adversarially by a separate model (Fable 5.1), six rounds so far. Findings are answered one by one in a response document. A finding counts as resolved only when the plan, the code and the pre-registrations agree.</p>
  <div class="tbl"><table><thead><tr><th>Round</th><th>Reviewed → revised</th><th>Findings</th><th>Verdict</th><th>What changed</th></tr></thead><tbody>
    <tr><td class="num">1</td><td class="num">v1 → v2</td><td class="num">21</td><td>Approve with changes</td><td>Hub choice and go/no-go gates, determinism levels split into fresh and replay parity, receipts, pilot designs</td></tr>
    <tr><td class="num">2</td><td class="num">v2 → v3</td><td class="num">19</td><td>Approve with changes</td><td>D2-replay relabelled as a diagnostic, fallback ladder with an acceptance test, merged checkpoints as certification reference</td></tr>
    <tr><td class="num">3</td><td class="num">v3 → v4</td><td class="num">15</td><td>Approve with changes</td><td>Architecture converged; pilot hygiene: proxy calibration, surface-form scoring, template clock, three-branch H4 rule</td></tr>
    <tr><td class="num">4</td><td class="num">v5 → v6</td><td class="num">21</td><td>Approve with changes</td><td>Feasibility model, served artifact A/B, G0 gates, agentic spike, allocation tiers, joint calibration, guards</td></tr>
    <tr><td class="num">5</td><td class="num">v6 → v6.1</td><td class="num">15</td><td>Approve with changes</td><td>Per-suite cost targets, δ_b from the tested difference, every-variant bound, roadmap order, goal-suite size, spike resolution</td></tr>
    <tr><td class="num">6</td><td class="num">v6.1 → v6.2</td><td class="num">7</td><td>Approve with changes</td><td>No blocker; structurally converged. Spike gate power, δ_b cap and pooling, switch rule on the spike checkpoint, confirmations that gate</td></tr>
    <tr><td class="num">7</td><td class="num">v6.2</td><td class="num">—</td><td>In progress</td><td>—</td></tr>
  </tbody></table></div>
  <div class="cols">
    <div><h3>Wave 1 research (determinism)</h3><ul>
      <li>Inference numerical determinism and batch invariance</li><li>MoE routing and candidate models</li><li>MoE interpretability and circuits</li><li>Prompt-to-DSL and constrained decoding</li><li>Task-specialised models</li><li>Routing, orchestration and harnesses</li><li>Consistency training and verification</li></ul></div>
    <div><h3>Wave 2 research (goal-compiled MoE)</h3><ul>
      <li>Groq LPU and compile-time scheduling</li><li>Token-efficient reasoning, System-1 and System-2</li><li>MoE expert expansion and router training</li><li>Ensembles, System-1 models and classic ML</li><li>Closing the gap: agentic training</li></ul></div>
    <div><h3>Primary evidence</h3><ul>
      <li>The user's A3B training report (52 pages), cited as A3B-§ or A3B-ch.N</li>
      <li>Model cards for Qwen3.6-35B-A3B, Qwen3.8-27B and Qwen3.8-Flash-Next</li>
      <li>CPU experiments 00–07 with pre-registrations and JSON results</li>
      <li>83 unit tests, including a test that every file and symbol the plan cites exists</li></ul></div>
  </div>
</section>"""

    body = "".join(s for _, _, s in sheets)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GCAM Implementation Blueprint</title>
<style>{font_css}</style><style>{style}</style><style>{PRINT_CSS}</style></head>
<body><div class="wrap" style="max-width:none;padding:0">{cover}{contents}{exec_summary}{body}{app_a}{app_b}{app_c}</div></body></html>"""


EXTRA = {
    "s1": """
  <h3 class="sub">Component specification</h3>
  <div class="tbl"><table><thead><tr><th>Component</th><th>Responsibility</th><th>Where it lives</th><th>Contract</th></tr></thead><tbody>
    <tr><td>Parse</td><td>Natural language → Goal IR; ambiguity becomes Clarify, never a guess</td><td>Dense parser spoke (P2); rule parser v0 in <code>dmc/compiler.py</code></td><td>D1 target</td></tr>
    <tr><td>Planner</td><td>Goal IR → DAG of typed ops; plans cached by template hash</td><td>Hub Plan profile (G1)</td><td>D0</td></tr>
    <tr><td>Scheduler</td><td>Ladders, budgets, caps, lanes, guards, variants, bound, schedule hash</td><td><code>dmc/scheduler.py</code></td><td>Pure function of IR, lock and cost models</td></tr>
    <tr><td>Executor registry</td><td>Cost models and per-item calibration records per executor and lock</td><td>T6 output; <code>Executor</code>, <code>Outcome</code></td><td>Versioned per lock</td></tr>
    <tr><td>Runtime</td><td>Runs the schedule, enforces caps, evaluates guards, records activities and receipts</td><td>Durable executor (P3, Temporal-class); P1 is the deterministic serving lane</td><td>Records every call</td></tr>
    <tr><td>System-1 tier</td><td>Code, classic ML, encoders, decision heads, SLM caller</td><td>T6</td><td>D1 on certified stacks</td></tr>
    <tr><td>Hub</td><td>Reasoning, planning, tool calls, creation, with goal-family capacity</td><td>Qwen3.6-35B-A3B after T1–T5</td><td>D0 + statistical D2 until the GDN audit</td></tr>
    <tr><td>Spokes</td><td>Teacher, judge and hardest-node escalation; reward model; verifiers</td><td>Qwen3.8-27B; Skywork-Reward-V2-Qwen3-8B</td><td>D0</td></tr>
    <tr><td>Trust layer</td><td>Lock, canonical hashing, result store, receipts, routing traces</td><td><code>dmc/lockfile.py</code>, <code>canonical.py</code>, <code>store.py</code>, <code>routing.py</code></td><td>Defines D0–D⊥</td></tr>
  </tbody></table></div>""",
    "s5": """
  <h3 class="sub">Hub specification</h3>
  <div class="tbl"><table><thead><tr><th>Property</th><th>Value</th><th>Source</th></tr></thead><tbody>
    <tr><td>Parameters</td><td>35B total, ~3B active per token</td><td>Qwen3.6-35B-A3B model card (W2-05)</td></tr>
    <tr><td>Layers</td><td>40 = 10 × (3 Gated DeltaNet + 1 gated attention), each followed by an MoE block; hidden size 2048</td><td>Model card, config (W2-03, W2-05)</td></tr>
    <tr><td>Experts</td><td>256 routed (top-8) + 1 shared; expert FFN width 512</td><td>Config (W2-03)</td></tr>
    <tr><td>One expert</td><td>3 × 2048 × 512 ≈ 3.15M parameters per layer; ≈ 126M across 40 layers (≈ 252 MB bf16)</td><td>Computed from the config (W2-03)</td></tr>
    <tr><td>Rank-16 LoRA on one expert</td><td>≈ 0.09–0.12M parameters; ~97 hot-expert LoRAs per layer ≈ 3–4 full experts</td><td>W2-03</td></tr>
    <tr><td>Goal experts at k = 48 (Artifact B)</td><td>≈ 6.0B parameters (≈ 6 GB FP8); 304 experts per layer = 8 × 38 for EP 8</td><td>Plan §I.2.3</td></tr>
    <tr><td>Vocabulary</td><td>248,320 tokens, shared with Qwen3.8-27B (full-logit distillation is possible)</td><td>Model cards (W2-05)</td></tr>
    <tr><td>Context; MTP</td><td>262,144 tokens native; multi-step MTP head (speedup on GDN unmeasured, G0-5)</td><td>Model card (W2-05)</td></tr>
    <tr><td>Checkpoint size</td><td>~72 GB bf16, ~36 GB block-FP8 after merge</td><td>A3B report ch.2</td></tr>
  </tbody></table></div>""",
}


def extract(src: str) -> tuple[str, list[tuple[str, str, str]]]:
    style = re.search(r"<style>(.*?)</style>", src, re.S).group(1)
    sheets = []
    for m in re.finditer(r'<section class="sheet" id="(s\d+)">(.*?)</section>', src, re.S):
        sid, inner = m.group(1), m.group(2)
        title = re.sub(r"<[^>]+>", "", re.search(r"<h2>(.*?)</h2>", inner, re.S).group(1)).strip()
        inner = f'<span class="marker">@@{sid}@@</span>' + inner + EXTRA.get(sid, "")
        sheets.append((sid, title, f'<section class="sheet" id="{sid}">{inner}</section>'))
    return style, sheets


def render(html_path: Path, pdf_path: Path) -> None:
    script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch({{ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' }});
  const page = await browser.newPage();
  await page.emulateMedia({{ media: 'print', colorScheme: 'light' }});
  await page.goto('file://{html_path}', {{ waitUntil: 'networkidle' }});
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({{
    path: '{pdf_path}', format: 'A4', landscape: true, printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true, outline: true, tagged: true,
    headerTemplate: '<div></div>',
    footerTemplate: `<div style="width:100%;padding:0 14mm;font-family:'DejaVu Sans',sans-serif;font-size:7px;color:#55667C;display:flex;justify-content:space-between;">
      <span>GCAM implementation blueprint · plan {PLAN_VERSION} · {DATE}</span>
      <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>`,
  }});
  await browser.close();
}})();
"""
    js = html_path.with_name("render.cjs")
    js.write_text(script)
    subprocess.run(["node", str(js)], check=True, env={"NODE_PATH": "/opt/node22/lib/node_modules", "PATH": "/opt/node22/bin:/usr/bin:/bin"})


def section_pages(pdf_path: Path) -> dict[str, int]:
    pages = {}
    for i, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        for key in re.findall(r"@@(\w+)@@", page.extract_text() or ""):
            pages.setdefault(key, i)
    return pages


def main() -> None:
    font_dir = Path(sys.argv[1]).resolve()
    if not (font_dir / "local.css").exists():
        fetch_fonts(font_dir)
    style, sheets = extract(SRC.read_text(encoding="utf-8"))
    font_css = (font_dir / "local.css").read_text()
    work = font_dir.parent
    html_path = work / "blueprint_print.html"
    tmp_pdf = work / "pass1.pdf"
    # Fonts are referenced relative to the HTML file, so the HTML sits next to the font dir.
    font_css = font_css.replace("src: url(", f"src: url({font_dir.name}/")

    html_path.write_text(build_html(sheets, style, font_css, {}).replace("{PAGES}", "—"))
    render(html_path, tmp_pdf)
    pages = section_pages(tmp_pdf)
    total = len(PdfReader(str(tmp_pdf)).pages)
    html_path.write_text(build_html(sheets, style, font_css, pages).replace("{PAGES}", str(total)))
    render(html_path, tmp_pdf)
    check = section_pages(tmp_pdf)
    if check != pages:
        raise SystemExit(f"page numbers moved between passes: {pages} vs {check}")

    writer = PdfWriter(clone_from=str(tmp_pdf))
    writer.add_metadata({
        "/Title": "GCAM Implementation Blueprint",
        "/Subject": f"Goal-Compiled Agentic MoE: architecture, training and implementation plan {PLAN_VERSION} as an 11-sheet drawing set",
        "/Keywords": "MoE, Qwen3.6-35B-A3B, compile-time scheduling, determinism, goal-gated experts, System-1",
        "/Creator": "docs/pdf/build_blueprint_pdf.py (Chromium, Playwright)",
    })
    with open(OUT, "wb") as f:
        writer.write(f)
    print(json.dumps({"pages": total, "sections": pages, "out": str(OUT)}))


if __name__ == "__main__":
    main()
