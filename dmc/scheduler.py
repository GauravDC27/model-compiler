"""Compile-time scheduling of a goal's task graph (Groq-style static scheduling, for agents).

Groq's LPU compiler decides, before execution, where every operation runs and when; the
hardware has no dynamic scheduler, so latency and results are known in advance. This module
applies the same idea one level up: a goal is compiled to a DAG of IR ops, and the scheduler
decides, before any model runs,
  * which executor runs each node (System-1 classifier, hub profile, specialist spoke, tool,
    classic ML model) — the cheapest ladder whose calibrated accuracy meets the node's bar
    and whose determinism contract meets the node's requirement;
  * the escalation ladder when a cheap executor abstains (a cascade with a static length);
  * every node's token budget, the schedule's expected tokens and its *worst-case* tokens,
    including compiled caps on tool-output input;
  * the lane and start slot of every node (list scheduling, critical path first), with nodes
    that share a prompt prefix on the same executor placed back-to-back so it is computed once.
Data-dependent control flow is handled by guards: each guard names a pre-compiled variant (or
a bounded recompile); the program's worst case covers every variant and the recompile
allowance, and an exhausted allowance yields Abstain rather than a loop.

Ladder accuracy is estimated, when a joint calibration set is supplied, by replaying the
ladder on those items (so errors that are correlated across rungs are accounted for); without
one it falls back to treating rungs as independent, and the schedule says so.

The schedule is a pure function of (graph, executors, calibration, budget, lock) and is
content-addressed. A graph whose worst case exceeds the budget is a compile error.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .canonical import content_hash

Kind = Literal["system1", "system2", "spoke", "tool", "classic_ml"]
# Determinism contract an executor can deliver on its certified stack (PLAN Part II):
# D0 = replay only (recorded result), D1 = bitwise recompute. Ordered weakest to strongest.
CONTRACTS = ("D0", "D1")
# Capability lattice for actions a node may take (fixed per node and branch at compile time).
CAPABILITIES = ("read", "write", "external_send", "destructive")


@dataclass(frozen=True)
class Executor:
    """A calibrated way to run some op types, with its cost model.

    ``coverage`` is the share of inputs it answers itself at its calibrated abstention
    threshold; ``accuracy`` is its accuracy on the inputs it answers (per op type);
    ``accuracy_all`` is its accuracy when it may not abstain (needed for it to end a ladder;
    an abstaining executor without it can only be a lower rung). These summary numbers are
    used only when no joint calibration set is supplied.

    Token fields are compiled sizes and budgets, so they bound a call: ``tokens_in`` is the
    fixed prompt (template, instructions; not the node's input), ``tokens_out`` the decode
    budget, and ``input_reads`` how many times the node's capped input enters the prompt per
    call (0 for executors that never feed it to a model, such as code; 2 for order-averaged
    scoring over two label orders). Expected tokens come from joint calibration records, which
    hold what calls really spent; without them the estimate falls back to these bounds.
    """

    name: str
    kind: Kind
    ops: frozenset[str]
    tokens_in: int  # fixed prompt tokens per call, excluding the node's input
    tokens_out: int  # generated-token budget per call (thinking + answer)
    latency: int  # integer time units per call
    accuracy: dict[str, float]
    coverage: float = 1.0
    accuracy_all: dict[str, float] | None = None
    contract: str = "D0"
    input_reads: int = 1
    version: str = "0"

    def terminal_accuracy(self, op: str) -> float:
        if self.accuracy_all is not None:
            return self.accuracy_all.get(op, 0.0)
        return self.accuracy.get(op, 0.0) if self.coverage >= 1.0 else 0.0

    def call_tokens(self, input_cap: int, discount: int = 0) -> int:
        llm_in = self.tokens_in + self.input_reads * input_cap
        return max(0, llm_in - discount) + self.tokens_out


@dataclass(frozen=True)
class Outcome:
    """One executor's behaviour on one calibration item."""

    answered: bool  # answered at its calibrated threshold (did not abstain)
    correct: bool  # correct when answering at threshold
    correct_forced: bool  # correct when forced to answer
    tokens: int  # tokens it spent on this item


# op (or "op#difficulty") -> one dict per calibration item: executor name -> Outcome
Calibration = Mapping[str, Sequence[Mapping[str, Outcome]]]


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    deps: tuple[str, ...] = ()
    required_accuracy: float = 0.0
    prefix_group: str | None = None  # nodes sharing a prompt prefix (e.g. same document)
    prefix_tokens: int = 0  # length of that shared prefix
    requires_contract: str = "D0"  # weakest determinism contract acceptable for this node
    input_cap: int = 0  # compiled cap on tool-output tokens fed to LLM rungs (runtime truncates)
    difficulty: str | None = None  # bucket from a compile-time probe; selects calibration
    capabilities: frozenset[str] = frozenset()  # actions this node may take


@dataclass(frozen=True)
class Assignment:
    node: str
    ladder: tuple[str, ...]  # executors tried in order; each escalates on abstention
    expected_accuracy: float
    expected_tokens: float
    worst_case_tokens: int
    lane: int
    start: int
    finish: int  # worst-case finish (every rung of the ladder runs)
    estimate: Literal["joint", "independent"] = "independent"


@dataclass(frozen=True)
class Schedule:
    assignments: tuple[Assignment, ...]
    expected_tokens: float
    worst_case_tokens: int
    makespan: int  # worst-case
    hash: str = field(default="")


class CompileError(ValueError):
    """The goal cannot be scheduled within its contract (accuracy, budget, or graph shape)."""


def _topo_order(nodes: Sequence[Node]) -> list[Node]:
    by_id = {n.id: n for n in nodes}
    if len(by_id) != len(nodes):
        raise CompileError("duplicate node ids")
    for n in nodes:
        for d in n.deps:
            if d not in by_id:
                raise CompileError(f"node {n.id!r} depends on unknown node {d!r}")
    indeg = {n.id: len(n.deps) for n in nodes}
    users: dict[str, list[str]] = {n.id: [] for n in nodes}
    for n in nodes:
        for d in n.deps:
            users[d].append(n.id)
    ready = sorted(i for i, k in indeg.items() if k == 0)
    order = []
    while ready:
        i = ready.pop(0)
        order.append(by_id[i])
        for u in sorted(users[i]):
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)
                ready.sort()
    if len(order) != len(nodes):
        raise CompileError("task graph has a cycle")
    return order


# (accuracy, expected tokens, share of requests reaching each rung)
Estimate = tuple[float, float, tuple[float, ...]]


def _estimate_independent(node: Node, ladder: Sequence[Executor]) -> Estimate:
    acc, cost, reach, reaches = 0.0, 0.0, 1.0, []
    for i, e in enumerate(ladder):
        last = i == len(ladder) - 1
        reaches.append(reach)
        acc += reach * (e.terminal_accuracy(node.op) if last else e.coverage * e.accuracy.get(node.op, 0.0))
        cost += reach * e.call_tokens(node.input_cap)
        reach *= 0.0 if last else 1.0 - e.coverage
    return acc, cost, tuple(reaches)


def _estimate_joint(node: Node, ladder: Sequence[Executor], items) -> Estimate | None:
    """Replay the ladder on calibration items: exact under correlated errors across rungs.

    Tokens are the ones each executor actually spent on the items that reached it, so a
    correlation between input length and escalation is priced in too.
    """
    if not items or any(e.name not in items[0] for e in ladder):
        return None
    correct, cost, reached = 0, 0.0, [0] * len(ladder)
    for rec in items:
        for i, e in enumerate(ladder):
            o = rec[e.name]
            reached[i] += 1
            cost += o.tokens
            if i == len(ladder) - 1:
                correct += o.correct_forced
                break
            if o.answered:
                correct += o.correct
                break
    n = len(items)
    return correct / n, cost / n, tuple(r / n for r in reached)


def _ladder(node: Node, executors: Sequence[Executor], max_rungs: int,
            calibration: Calibration | None) -> tuple[list[Executor], Estimate, str]:
    """Cheapest cascade meeting the node's accuracy bar.

    A ladder is any cost-ordered subsequence of the capable executors (so a System-1 model may
    escalate straight to a specialist), at most ``max_rungs`` long; every rung but the last
    may abstain. Ties break on executor names, so the choice is deterministic.
    """
    need = CONTRACTS.index(node.requires_contract)
    cands = sorted(
        (e for e in executors if node.op in e.ops and CONTRACTS.index(e.contract) >= need),
        key=lambda e: (e.call_tokens(node.input_cap), e.latency, e.name),
    )
    if not cands:
        raise CompileError(
            f"no executor can run op {node.op!r} at contract {node.requires_contract} (node {node.id!r})"
        )
    key = node.op if node.difficulty is None else f"{node.op}#{node.difficulty}"
    items = (calibration or {}).get(key)
    best: tuple[float, list[str], list[Executor], float, str] | None = None
    top_acc = 0.0
    for size in range(1, min(max_rungs, len(cands)) + 1):
        for ladder in itertools.combinations(cands, size):
            joint = _estimate_joint(node, ladder, items)
            est = joint if joint is not None else _estimate_independent(node, ladder)
            acc, cost, _ = est
            top_acc = max(top_acc, acc)
            if acc + 1e-12 < node.required_accuracy:
                continue
            names = [e.name for e in ladder]
            if best is None or (cost, names) < (best[0], best[1]):
                best = (cost, names, list(ladder), est, "joint" if joint is not None else "independent")
    if best is None:
        raise CompileError(
            f"node {node.id!r} needs accuracy {node.required_accuracy} but the best ladder "
            f"for {node.op!r} reaches {round(top_acc, 6)}"
        )
    return best[2], best[3], best[4]


def compile_schedule(
    nodes: Sequence[Node],
    executors: Sequence[Executor],
    *,
    token_budget: int,
    lanes: int = 1,
    max_rungs: int = 3,
    calibration: Calibration | None = None,
    lock_hash: str = "",
) -> Schedule:
    order = _topo_order(nodes)
    plans = {n.id: _ladder(n, executors, max_rungs, calibration) for n in order}

    # A prefix is reused only by the same executor on the same lane (caches do not cross models).
    seen_prefix: set[tuple[int, str, str]] = set()

    users: dict[str, list[str]] = {n.id: [] for n in order}
    for n in order:
        for d in n.deps:
            users[d].append(n.id)
    node_lat = {i: sum(e.latency for e in plans[i][0]) for i in plans}
    rank: dict[str, int] = {}
    for n in reversed(order):
        rank[n.id] = node_lat[n.id] + max((rank[u] for u in users[n.id]), default=0)

    lane_free = [0] * lanes
    lane_group: list[str | None] = [None] * lanes
    finish: dict[str, int] = {}
    done: set[str] = set()
    pending = list(order)
    out: list[Assignment] = []
    total_exp, total_worst = 0.0, 0
    while pending:
        ready = [n for n in pending if all(d in done for d in n.deps)]
        # Highest critical path first; prefer continuing an open prefix group; then id.
        ready.sort(key=lambda n: (-rank[n.id], n.prefix_group is None, n.prefix_group or "", n.id))
        n = ready[0]
        earliest = max((finish[d] for d in n.deps), default=0)

        def lane_key(i: int) -> tuple:
            # Earliest start; on ties keep a prefix group on the lane that already holds it.
            same_group = n.prefix_group is not None and lane_group[i] == n.prefix_group
            return (max(lane_free[i], earliest), not same_group, i)

        lane = min(range(lanes), key=lane_key)
        start = max(lane_free[lane], earliest)
        ladder, (acc, exp_tokens, reaches), estimate = plans[n.id]
        worst = 0
        for e, reach in zip(ladder, reaches):
            key = (lane, n.prefix_group or "", e.name)
            discount = n.prefix_tokens if (n.prefix_group is not None and key in seen_prefix) else 0
            if n.prefix_group is not None and e.tokens_in > 0:
                seen_prefix.add(key)
            # The estimate priced every call at full prompt; credit the cached prefix.
            exp_tokens -= reach * (e.call_tokens(n.input_cap) - e.call_tokens(n.input_cap, discount))
            worst += e.call_tokens(n.input_cap, discount)
        end = start + node_lat[n.id]
        lane_free[lane], lane_group[lane] = end, n.prefix_group
        finish[n.id] = end
        done.add(n.id)
        pending.remove(n)
        total_exp += exp_tokens
        total_worst += worst
        out.append(Assignment(n.id, tuple(e.name for e in ladder), round(acc, 6),
                              round(exp_tokens, 3), worst, lane, start, end, estimate))

    if total_worst > token_budget:
        raise CompileError(
            f"worst-case tokens {total_worst} exceed the budget {token_budget}; "
            "lower accuracy bars, add cheaper executors, or raise the budget"
        )
    used = sorted({name for a in out for name in a.ladder})
    versions = {e.name: (e.version, e.contract) for e in executors if e.name in used}
    h = content_hash({
        "assignments": [a.__dict__ for a in out], "budget": token_budget, "lanes": lanes,
        "executors": versions, "lock": lock_hash,
    })
    return Schedule(tuple(out), round(total_exp, 3), total_worst, max(finish.values(), default=0), h)


# --------------------------------------------------------------------------------------------
# Programs: guards, pre-compiled variants and a bounded recompile allowance.


@dataclass(frozen=True)
class Guard:
    """A typed predicate checked after ``node``; on failure, run ``on_fail``.

    ``predicate`` names an IR-level check over the node's typed output (e.g. "tests_pass",
    "nonempty_result"); it is evaluated over data, never by free-text model judgement.
    ``on_fail`` is a variant id, or "recompile" to spend one unit of the recompile allowance.
    """

    name: str
    node: str
    predicate: str
    on_fail: str


@dataclass(frozen=True)
class Program:
    main: tuple[Node, ...]
    variants: Mapping[str, tuple[Node, ...]] = field(default_factory=dict)
    guards: tuple[Guard, ...] = ()
    recompile_allowance: int = 0  # how many recompiles may happen at runtime
    recompile_budget: int = 0  # worst-case tokens reserved per recompile
    capability_ceiling: frozenset[str] = frozenset({"read"})


@dataclass(frozen=True)
class ProgramSchedule:
    main: Schedule
    variants: Mapping[str, Schedule]
    worst_case_tokens: int  # main + most expensive variant chain + recompile reserve
    hash: str


def compile_program(program: Program, executors: Sequence[Executor], *, token_budget: int,
                    lanes: int = 1, calibration: Calibration | None = None,
                    lock_hash: str = "") -> ProgramSchedule:
    all_nodes = list(program.main) + [n for v in program.variants.values() for n in v]
    for n in all_nodes:
        extra = n.capabilities - program.capability_ceiling
        if extra:
            raise CompileError(f"node {n.id!r} needs capabilities {sorted(extra)} beyond the ceiling")
        unknown = n.capabilities - set(CAPABILITIES)
        if unknown:
            raise CompileError(f"node {n.id!r} has unknown capabilities {sorted(unknown)}")
    main_ids = {n.id for n in program.main}
    for g in program.guards:
        if g.node not in main_ids and not any(g.node in {n.id for n in v} for v in program.variants.values()):
            raise CompileError(f"guard {g.name!r} watches unknown node {g.node!r}")
        if g.on_fail != "recompile" and g.on_fail not in program.variants:
            raise CompileError(f"guard {g.name!r} targets unknown variant {g.on_fail!r}")
    if any(g.on_fail == "recompile" for g in program.guards) and program.recompile_allowance == 0:
        raise CompileError("a guard may recompile but the recompile allowance is 0")

    kw = dict(lanes=lanes, calibration=calibration, lock_hash=lock_hash, token_budget=10**18)
    main = compile_schedule(program.main, executors, **kw)
    variants = {vid: compile_schedule(nodes, executors, **kw) for vid, nodes in sorted(program.variants.items())}
    # Worst case: the main graph, plus the most expensive variant (variants replace the work
    # that follows a failed guard; counting them on top is conservative), plus the reserve.
    worst = (main.worst_case_tokens + max((v.worst_case_tokens for v in variants.values()), default=0)
             + program.recompile_allowance * program.recompile_budget)
    if worst > token_budget:
        raise CompileError(f"program worst-case tokens {worst} exceed the budget {token_budget}")
    h = content_hash({"main": main.hash, "variants": {k: v.hash for k, v in variants.items()},
                      "guards": [g.__dict__ for g in program.guards],
                      "recompile": [program.recompile_allowance, program.recompile_budget],
                      "ceiling": sorted(program.capability_ceiling), "budget": token_budget})
    return ProgramSchedule(main, variants, worst, h)


def cap_input(ids: Sequence[int], cap: int, marker: Sequence[int]) -> tuple[list[int], bool]:
    """Truncate a tool output to its compiled cap: keep the head and the tail around a marker.

    A pure function of its arguments, so replay truncates identically. Returns the tokens to
    feed and whether the cap was hit (recorded in ``Receipt.cap_hits``).
    """
    if len(ids) <= cap:
        return list(ids), False
    room = cap - len(marker)
    if room < 2:
        raise ValueError(f"cap {cap} leaves no room around a {len(marker)}-token marker")
    head = room - room // 2
    tail = room // 2
    return list(ids[:head]) + list(marker) + list(ids[len(ids) - tail:]), True


class GuardRuntime:
    """Runtime side of guards: decides what happens when a guard fails."""

    def __init__(self, program: Program):
        self.program = program
        self.guards = {g.name: g for g in program.guards}
        self.recompiles_used = 0
        self.variants_used: list[str] = []

    def on_failure(self, guard_name: str) -> str:
        """Returns a variant id, "recompile", or "abstain" when the allowance is exhausted."""
        g = self.guards[guard_name]
        if g.on_fail != "recompile":
            if g.on_fail in self.variants_used:  # a variant runs at most once: no silent loops
                return "abstain"
            self.variants_used.append(g.on_fail)
            return g.on_fail
        if self.recompiles_used >= self.program.recompile_allowance:
            return "abstain"
        self.recompiles_used += 1
        return "recompile"
