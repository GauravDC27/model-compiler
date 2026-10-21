"""Assemble GCAM v1: every executor, its cost model, its calibration records, and the lock.

An executor *rung* is one (implementation, op) pair registered with the scheduler as a
``dmc.scheduler.Executor`` whose token fields are compiled bounds in hub-token equivalents
(``executors.base.COST_WEIGHT``). Expected cost and accuracy come from the joint calibration
records written by ``gcam.train.calibrate``; the schedule hash includes the lock hash, so any
change to a model, adapter, threshold or record set invalidates every compiled schedule.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers

from dmc.canonical import content_hash, file_hash
from dmc.scheduler import Executor, Outcome

from .domain import World, build_world
from .executors.base import COST_WEIGHT
from .executors.classic import OrderForecaster
from .executors.code import EffectLog, build_code_executors
from .executors.llm import LLMExecutor
from .models import HUB_ID, SPOKE_ID, LM, Capacity, artifacts_dir

ALIAS = {
    "decide.claim_reason": "claim", "tool.select": "tool", "evaluate.pairwise": "pair", "create.reply": "reply",
    "create.reply_fallback": "reply2", "create.summary": "summary", "analyze.estimate": "estimate",
    "decide.expense_review": "expense", "retrieve.policy": "policy",
}
# Ladder ops and the implementations that may serve them (cheapest first is the scheduler's job).
CANDIDATES = {
    "decide.claim_reason": ("code.keywords", "s1.encoder", "s1.hubprobe", "hub", "hub.policy", "spoke"),
    "tool.select": ("code.keywords", "s1.encoder", "hub", "hub.support", "spoke"),
    "evaluate.pairwise": ("code.verify", "hub", "spoke"),
    "create.reply": ("hub",),  # the spoke writes only in the pre-compiled fallback variant
    "create.reply_fallback": ("spoke",),
    "create.summary": ("hub", "spoke"),
    "analyze.estimate": ("code.arith", "hub", "spoke"),
    "decide.expense_review": ("ml.gbdt", "hub", "spoke"),
    "retrieve.policy": ("ml.tfidf", "s1.encoder", "hub", "spoke"),
}
# Ops with one exact executor (compiled code or a deterministic model): accuracy 1 by contract.
EXACT = {
    "tool.get_order": "code.lookup", "decide.refund": "code.policy", "tool.args": "code.args",
    "meta.confirm": "code.confirm", "tool.execute": "code.tools", "analyze.refund_rates": "code.analytics",
    "create.summary_template": "code.analytics", "analyze.collect": "code.analytics",
    "create.policy_answer": "code.analytics", "forecast.orders": "ml.forecast",
}
LATENCY = {"code": 0, "ml": 1, "s1": 1, "hub": 4, "spoke": 12}
GENERATIVE = {"create.reply": 3 * 70, "create.reply_fallback": 70, "create.summary": 90, "analyze.estimate": 48}


@dataclass
class Rung:
    name: str
    impl: object
    op: str
    executor: Executor


def rung_name(impl: str, op: str) -> str:
    return f"{impl}[{ALIAS.get(op, op)}]"


def _family_of(impl: str) -> str:
    return impl.split(".")[0]


class System:
    """Everything a goal needs at run time. ``llms=False`` builds the LLM-free subset (tests)."""

    def __init__(self, *, llms: bool = True, art: Path | None = None, world: World | None = None):
        self.art = art or artifacts_dir()
        self.world = world or build_world()
        self.effects = EffectLog()
        self.impls: dict[str, object] = dict(build_code_executors(self.world, self.effects))
        self.impls["ml.forecast"] = OrderForecaster(self.world.daily_orders)
        self.models = {}
        cal_path = self.art / "calibration.json"
        self.calibration = json.loads(cal_path.read_text()) if cal_path.exists() else {"records": {}, "thresholds": {}}
        sk = self.art / "system1.pkl"
        if sk.exists():
            with open(sk, "rb") as f:
                saved = pickle.load(f)
            self.impls["ml.gbdt"] = saved["gbdt"]
            self.impls["ml.tfidf"] = saved["tfidf"]
        if llms:
            self._load_llms(saved if sk.exists() else None)
        self.rungs = self._register()

    # ---------------------------------------------------------------- models
    def _load_llms(self, saved) -> None:
        from .executors.system1 import Encoder, EncoderHeads, HubProbe

        hub = LM(HUB_ID, "hub")
        cap = Capacity(hub)
        hub.capacity = cap
        manifest = json.loads((self.art / "capacity" / "manifest.json").read_text())
        parents = {int(k): v for k, v in manifest["families"]["policy"]["parents"].items()}
        cap.add_goal_experts("policy", parents, variant=manifest["families"]["policy"].get("variant", "routed"))
        hot = {int(k): v for k, v in manifest["families"]["support"]["hot"].items()}
        cap.add_lora("support", hot, rank=manifest["families"]["support"]["rank"])
        for fam in ("policy", "support"):
            cap.load(fam, torch.load(self.art / "capacity" / f"{fam}.pt"))
        spoke = LM(SPOKE_ID, "spoke")
        self.models = {"hub": hub, "spoke": spoke}
        thresholds = self.calibration.get("thresholds", {})
        for name, lm, fam in (("hub", hub, {}), ("hub.policy", hub, {"decide.claim_reason": "policy"}),
                              ("hub.support", hub, {"tool.select": "support"}), ("spoke", spoke, {})):
            ex = LLMExecutor(name, lm, family=fam)
            ex.tau = {op: t for key, t in thresholds.items() for op in [key.split("|")[1]] if key.split("|")[0] == name}
            self.impls[name] = ex
        if saved is not None:
            enc = Encoder()
            self.models["encoder"] = enc
            self.impls["s1.encoder"] = EncoderHeads(enc, saved["encoder_heads"], saved["retrieval_tau"])
            self.impls["s1.hubprobe"] = HubProbe(hub, saved["probe_layer"], saved["probe_heads"])

    # ---------------------------------------------------------------- registry
    def _samples(self) -> dict[str, dict]:
        """Representative payloads per op, used to measure each prompt template's fixed size."""
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in self.world.expenses.iloc[0].to_dict().items()}
        facts = self.impls["code.analytics"].run("analyze.refund_rates", {"days": 180, "now": "2026-10-21T10:00:00+00:00"},
                                                 forced=True).value
        reply = {"customer_name": "Leila Petrov-Nasser", "order_id": "A-5000", "product": "kitchen knife set",
                 "reason": "not_received", "decision": "deny", "rule": "P3"}
        return {"decide.claim_reason": {"text": ""}, "tool.select": {"text": ""}, "retrieve.policy": {"query": ""},
                "decide.expense_review": {"row": {**row, "vendor": "FlowerPost", "category": "software",
                                                  "amount": 99999.99, "tenure_months": 119}},
                "evaluate.pairwise": {"a": "", "b": "", "customer_name": "Leila Petrov-Nasser", "order_id": "A-5000"},
                "create.reply": reply, "create.reply_fallback": reply, "create.summary": {"facts": facts},
                "analyze.estimate": {"text": ""}}

    def _bounds(self, impl: str, op: str) -> tuple[int, int, int]:
        """(fixed prompt, input reads, output budget) of one call, in hub-token equivalents.

        Fixed = the op's prompt template(s) measured with the capped input blanked, plus 8 tokens
        of boundary slack per prompt; reads = how many prompts carry the capped input. Rounded up,
        so they are bounds on what a call can cost once the runtime has truncated the input."""
        fam = _family_of(impl)
        if fam in ("code", "ml"):
            return 0, 0, 0
        w = COST_WEIGHT["spoke" if impl == "spoke" else "encoder" if impl == "s1.encoder" else "hub"]
        if impl in ("s1.encoder", "s1.hubprobe"):
            return 0, math.ceil(w), 0
        from .executors.llm import prompts
        calls = len(prompts(op, self._sample[op], self.impls[impl].k))
        fixed = self.impls[impl].template_tokens(op, self._sample[op]) + 8 * calls
        reads = calls if op not in ("create.reply", "create.reply_fallback", "create.summary") else 0
        return math.ceil(w * fixed), math.ceil(w * reads), math.ceil(w * GENERATIVE.get(op, 0))

    def _register(self) -> dict[str, Rung]:
        rungs = {}
        self._sample = self._samples()
        recs = self.calibration.get("records", {})
        for op, impls in CANDIDATES.items():
            for impl in impls:
                if impl not in self.impls:
                    continue
                name = rung_name(impl, op)
                fixed, reads, out = self._bounds(impl, op)
                acc, cov, acc_all = _summary(recs.get(op), name)
                ex = Executor(name, "spoke" if impl == "spoke" else "system2" if impl.startswith("hub") else
                              "tool" if impl.startswith("code") else "system1", frozenset({op}), fixed, out,
                              LATENCY[_family_of(impl)] if _family_of(impl) in LATENCY else 1, {op: acc},
                              coverage=cov, accuracy_all={op: acc_all}, contract="D1", input_reads=reads,
                              version=getattr(self.impls[impl], "version", "1"))
                rungs[name] = Rung(name, self.impls[impl], op, ex)
        for op, impl in EXACT.items():
            if impl in self.impls:
                name = rung_name(impl, op)
                rungs[name] = Rung(name, self.impls[impl], op, Executor(
                    name, "tool", frozenset({op}), 0, 0, 0, {op: 1.0}, coverage=1.0, contract="D1", input_reads=0,
                    version=getattr(self.impls[impl], "version", "1")))
        return rungs

    @property
    def executors(self) -> list[Executor]:
        return [r.executor for r in self.rungs.values()]

    def outcomes(self) -> dict[str, list[dict[str, Outcome]]]:
        """Joint calibration records as the scheduler consumes them (op -> items -> rung -> Outcome)."""
        out = {}
        for op, items in self.calibration.get("records", {}).items():
            out[op] = [{name: Outcome(**o) for name, o in it.items() if name in self.rungs} for it in items]
        return out

    # ---------------------------------------------------------------- lock
    def lock(self) -> dict:
        fp = self.art / "fingerprints.json"
        cached = json.loads(fp.read_text()) if fp.exists() else {}
        models = {}
        for key, lm in self.models.items():
            mid = getattr(lm, "model_id", None)
            if mid is None:
                continue
            if mid not in cached:
                from huggingface_hub import snapshot_download
                path = Path(snapshot_download(mid, local_files_only=True))
                cached[mid] = content_hash([file_hash(str(p)) for p in sorted(path.glob("*.safetensors"))])
            models[key] = {"id": mid, "weights": cached[mid]}
        fp.write_text(json.dumps(cached, indent=1))
        # Only what changes outputs: hub capacity, System-1/classic models, thresholds and records.
        files = [self.art / "system1.pkl", self.art / "calibration.json", *sorted((self.art / "capacity").glob("*"))]
        arts = {str(p.relative_to(self.art)): file_hash(str(p)) for p in files if p.is_file()}
        return {"schema": "gcam.lock/1", "models": models, "artifacts": arts, "world": self.world.snapshot_hash(),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "executors": {n: r.executor.version for n, r in sorted(self.rungs.items())}}

    def lock_hash(self) -> str:
        return content_hash(self.lock())


def _summary(items: list | None, name: str) -> tuple[float, float, float]:
    """Accuracy on answered items, coverage and forced accuracy from joint records (fallback numbers)."""
    rows = [it[name] for it in (items or []) if name in it]
    if not rows:
        return 0.0, 0.0, 0.0
    answered = [r for r in rows if r["answered"]]
    acc = sum(r["correct"] for r in answered) / len(answered) if answered else 0.0
    return acc, len(answered) / len(rows), sum(r["correct_forced"] for r in rows) / len(rows)
