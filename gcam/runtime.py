"""The GCAM v1 runtime: compile a goal once, then execute it exactly as compiled (plan §I.2.1).

``Runtime.run`` parses → plans → compiles (``dmc.scheduler.compile_program``: ladders from
joint calibration, contracts, capability ceiling, confirmation gating, worst-case bound,
schedule hash including the lock hash) → executes node by node:

* each node's capped inputs are truncated deterministically (``dmc.scheduler.cap_input``);
* its ladder runs rung by rung, escalating only when a rung abstains; the last rung is forced;
* every node writes a receipt to the content-addressed result store, so re-running the same
  goal on the same lock replays recorded bytes (D0) instead of recomputing;
* guards check typed outputs; on failure ``GuardRuntime`` returns a pre-compiled variant (run
  once, its output replaces the guarded node's), or ``abstain`` (a defined failure).

Nothing is re-planned at run time. Realized cost is checked against the compiled bound.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from dmc.canonical import canonical_json, content_hash
from dmc.scheduler import CompileError, GuardRuntime, ProgramSchedule, cap_input, compile_program
from dmc.store import Receipt, Record, ResultStore, result_key

from .executors.code import reply_checks, summary_checks
from .executors.llm import INPUT_FIELDS
from .goal_ir import Clarify, Goal
from .parser import parse
from .planner import GoalPlan, Planner
from .system import System

RUNTIME_VERSION = "gcam.runtime/1"
DEFAULT_BUDGET = 400_000  # hub-token equivalents per goal

GUARDS = {
    "order_found": lambda out, payload: out is not None,
    "args_complete": lambda out, payload: not out["missing"] and not out["invalid"],
    "approved": lambda out, payload: bool(out["approved"]),
    "reply_verified": lambda out, payload: all(reply_checks(out, payload["customer_name"], payload["order_id"],
                                                             payload["decision"]).values()),
    "summary_verified": lambda out, payload: all(summary_checks(out, payload["facts"]).values()),
}


@dataclass
class NodeTrace:
    node: str
    op: str
    ladder: tuple[str, ...]
    tried: list[dict]
    executor: str
    value: Any
    cost: int
    worst_case: int
    replayed: bool
    cap_hits: int
    receipt: str
    seconds: float


@dataclass
class GoalResult:
    text: str
    status: str  # done | abstained | clarify | compile_error
    family: str | None = None
    result: dict = field(default_factory=dict)
    nodes: list[NodeTrace] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    reason: str | None = None
    schedule_hash: str | None = None
    template_hash: str | None = None
    worst_case: int = 0
    cost: int = 0
    seconds: float = 0.0
    clarify: Clarify | None = None

    @property
    def bound_respected(self) -> bool:
        return self.cost <= self.worst_case


class Runtime:
    def __init__(self, system: System, store_dir=None, *, budget: int = DEFAULT_BUDGET):
        self.system = system
        self.planner = Planner()
        self.store = ResultStore(store_dir or system.art / "store")
        self.budget = budget
        self.lock_hash = system.lock_hash()
        self.outcomes = system.outcomes()
        self._compiled: dict[str, ProgramSchedule] = {}
        self.hub_tok = system.models["hub"].backend if "hub" in system.models else None

    # ---------------------------------------------------------------- compile
    def compile(self, goal: Goal) -> tuple[GoalPlan, ProgramSchedule]:
        plan = self.planner.plan(goal, self.system.world)
        key = f"{plan.template_hash}|{self.lock_hash}"
        if key not in self._compiled:
            self._compiled[key] = compile_program(plan.program, self.system.executors, token_budget=self.budget,
                                                  calibration=self.outcomes, lock_hash=self.lock_hash)
        return plan, self._compiled[key]

    # ---------------------------------------------------------------- execute
    def run(self, goal_or_text: Goal | str, *, autonomy: str = "confirm", approvals: tuple[str, ...] = ()) -> GoalResult:
        t0 = time.time()
        if isinstance(goal_or_text, str):
            parsed = parse(goal_or_text, self.system.world, autonomy=autonomy)
            if isinstance(parsed, Clarify):
                return GoalResult(goal_or_text, "clarify", reason=parsed.question, clarify=parsed,
                                  seconds=time.time() - t0)
            goal = parsed
        else:
            goal = goal_or_text
        if approvals:
            goal = goal.model_copy(update={"slots": {**goal.slots, "approvals": tuple(approvals)}})
        res = GoalResult(goal.text, "done", family=goal.family)
        try:
            plan, ps = self.compile(goal)
        except CompileError as e:
            res.status, res.reason, res.seconds = "compile_error", str(e), time.time() - t0
            return res
        res.schedule_hash, res.template_hash, res.worst_case = ps.hash, plan.template_hash, ps.worst_case_tokens
        outputs: dict[str, Any] = {}
        payloads: dict[str, dict] = {}
        guards = GuardRuntime(plan.program)
        by_node: dict[str, list] = {}
        for g in plan.program.guards:
            by_node.setdefault(g.node, []).append(g)
        replaces = plan.notes["replaces"]

        def execute(assignments) -> bool:
            for a in assignments:
                np_ = plan.nodes[a.node]
                payload = np_.build(goal, outputs, self.system.world)
                payload, hits = self._cap(np_.node.op, payload, np_.node.input_cap)
                payloads[a.node] = payload
                trace = self._run_node(goal, np_.node.id, np_.node.op, a.ladder, payload, a.worst_case_tokens, hits)
                res.nodes.append(trace)
                outputs[a.node] = trace.value
                if a.node in replaces:  # a variant node's output stands in for the node it replaces
                    outputs[replaces[a.node]] = trace.value
                for g in by_node.get(a.node, []):
                    if GUARDS[g.predicate](trace.value, payload):
                        continue
                    action = guards.on_failure(g.name)
                    res.events.append(f"guard {g.name} failed on {a.node} -> {action}")
                    if action == "abstain":
                        res.status, res.reason = "abstained", f"guard {g.name} failed on node {a.node}"
                        return False
                    if action in ps.variants and not execute(ps.variants[action].assignments):
                        return False
            return True

        ok = execute(ps.main.assignments)
        if ok:
            res.result = {nid: outputs.get(nid) for nid in plan.result}
        elif goal.family == "support_request" and "args" in outputs:
            res.result = {"tool": outputs.get("tool"), "args": outputs.get("args"), "confirm": outputs.get("confirm")}
        res.cost = sum(n.cost for n in res.nodes)
        res.seconds = time.time() - t0
        return res

    def _cap(self, op: str, payload: dict, cap: int) -> tuple[dict, int]:
        """Truncate the op's capped input fields to ``cap`` hub tokens (head + tail around a marker)."""
        if not cap or self.hub_tok is None or op not in INPUT_FIELDS:
            return payload, 0
        fields = INPUT_FIELDS[op]
        per = max(8, cap // len(fields))
        out = json.loads(json.dumps(payload))
        hits = 0
        marker = self.hub_tok._ids(" … ")
        for f in fields:
            holder, key = (out[f.split(".")[0]], f.split(".")[1]) if "." in f else (out, f)
            ids = self.hub_tok._ids(holder[key])
            capped, hit = cap_input(ids, per, marker)
            if hit:
                holder[key] = self.hub_tok.tok.decode(capped)
                hits += 1
        return out, hits

    def _run_node(self, goal: Goal, nid: str, op: str, ladder: tuple[str, ...], payload: dict, worst: int,
                  hits: int) -> NodeTrace:
        t0 = time.time()
        ir_hash = content_hash({"op": op, "payload": payload, "ladder": list(ladder)})
        key = result_key(self.lock_hash, ir_hash, RUNTIME_VERSION, None)
        rec = self.store.get(key, lock_hash=self.lock_hash)
        if rec is not None:  # D0: same canonical input + lock -> the recorded bytes
            return NodeTrace(nid, op, ladder, [], rec.receipt.profile, json.loads(rec.output), 0, worst, True, hits,
                             key, time.time() - t0)
        tried, res, used = [], None, ladder[-1]
        for i, name in enumerate(ladder):
            rung = self.system.rungs[name]
            res = rung.impl.run(rung.op, payload, forced=i == len(ladder) - 1)
            tried.append({"rung": name, "answered": res.answered, "confidence": None if res.confidence is None
                          else round(float(res.confidence), 4), "cost": res.cost})
            if res.answered:
                used = name
                break
        cost = sum(t["cost"] for t in tried)
        receipt = Receipt(key=key, lock_hash=self.lock_hash, ir_hash=ir_hash,
                          session_state_hash=goal.context.session_state_hash, profile=used,
                          profile_version=self.system.rungs[used].executor.version, seed=None,
                          margins={"confidence": float(res.confidence or 0.0)},
                          verifier={"ladder": ">".join(ladder), "escalations": str(len(tried) - 1)}, cap_hits=hits)
        self.store.put(Record(receipt=receipt, output=canonical_json(res.value)))
        return NodeTrace(nid, op, ladder, tried, used, res.value, cost, worst, False, hits, key, time.time() - t0)
