"""Compile-time scheduling of a goal's task graph (Groq-style static scheduling, for agents).

Groq's LPU compiler decides, before execution, where every operation runs and when; the
hardware has no dynamic scheduler, so latency and results are known in advance. This module
applies the same idea one level up: a goal is compiled to a DAG of IR ops, and the scheduler
decides, before any model runs,
  * which executor runs each node (System-1 classifier, hub profile, specialist spoke, tool,
    classic ML model) — the cheapest one whose calibrated accuracy meets the node's bar;
  * the escalation ladder when the cheap executor abstains (a cascade with a static length);
  * every node's token budget, the schedule's expected tokens and its *worst-case* tokens;
  * the lane and start slot of every node (list scheduling, critical path first), with nodes
    that share a prompt prefix placed back-to-back on one lane so the prefix is computed once.
The schedule is a pure function of (graph, executors, budget) and is content-addressed. A
graph whose worst case exceeds the budget is a compile error, not a runtime surprise.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .canonical import content_hash

Kind = Literal["system1", "system2", "spoke", "tool", "classic_ml"]


@dataclass(frozen=True)
class Executor:
    """A calibrated way to run some op types, with its cost model.

    ``coverage`` is the share of inputs it answers itself at its calibrated abstention
    threshold; ``accuracy`` is its accuracy on the inputs it answers (per op type);
    ``accuracy_all`` is its accuracy when it may not abstain (needed for it to end a ladder;
    an abstaining executor without it can only be a lower rung). All come from held-out
    evaluation and are versioned with the executor.
    """

    name: str
    kind: Kind
    ops: frozenset[str]
    tokens_in: int  # prompt tokens per call (0 for non-LLM executors)
    tokens_out: int  # generated-token budget per call (thinking + answer)
    latency: int  # integer time units per call
    accuracy: dict[str, float]
    coverage: float = 1.0
    accuracy_all: dict[str, float] | None = None
    version: str = "0"

    def terminal_accuracy(self, op: str) -> float:
        if self.accuracy_all is not None:
            return self.accuracy_all.get(op, 0.0)
        return self.accuracy.get(op, 0.0) if self.coverage >= 1.0 else 0.0


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    deps: tuple[str, ...] = ()
    required_accuracy: float = 0.0
    prefix_group: str | None = None  # nodes sharing a prompt prefix (e.g. same document)
    prefix_tokens: int = 0  # length of that shared prefix


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


def _ladder(node: Node, executors: Sequence[Executor], max_rungs: int) -> tuple[list[Executor], float]:
    """Cheapest cascade meeting the node's accuracy bar.

    A ladder is any cost-ordered subsequence of the capable executors (so a System-1 model may
    escalate straight to a specialist), at most ``max_rungs`` long; every rung but the last
    may abstain. Ties break on executor names, so the choice is deterministic.
    """
    cands = sorted(
        (e for e in executors if node.op in e.ops),
        key=lambda e: (e.tokens_in + e.tokens_out, e.latency, e.name),
    )
    if not cands:
        raise CompileError(f"no executor can run op {node.op!r} (node {node.id!r})")
    best: tuple[float, list[str], list[Executor], float] | None = None
    for size in range(1, min(max_rungs, len(cands)) + 1):
        for ladder in itertools.combinations(cands, size):
            acc, cost, reach = 0.0, 0.0, 1.0
            for i, e in enumerate(ladder):
                if i == len(ladder) - 1:
                    acc += reach * e.terminal_accuracy(node.op)
                else:
                    acc += reach * e.coverage * e.accuracy.get(node.op, 0.0)
                cost += reach * (e.tokens_in + e.tokens_out)
                reach *= 1.0 - (1.0 if i == len(ladder) - 1 else e.coverage)
            if acc + 1e-12 < node.required_accuracy:
                continue
            key = (cost, [e.name for e in ladder])
            if best is None or key < (best[0], best[1]):
                best = (cost, key[1], list(ladder), acc)
    if best is None:
        top_acc = max(e.terminal_accuracy(node.op) for e in cands)
        raise CompileError(
            f"node {node.id!r} needs accuracy {node.required_accuracy} but the best executor "
            f"for {node.op!r} reaches {top_acc}"
        )
    return best[2], best[3]


def compile_schedule(
    nodes: Sequence[Node],
    executors: Sequence[Executor],
    *,
    token_budget: int,
    lanes: int = 1,
    max_rungs: int = 3,
) -> Schedule:
    order = _topo_order(nodes)
    plans = {n.id: _ladder(n, executors, max_rungs) for n in order}

    # Prefix sharing: the first node of a group on a lane pays for the prefix, later ones do not.
    seen_prefix: set[tuple[int, str]] = set()

    # Critical-path priority (worst-case latency to the end of the graph).
    users: dict[str, list[str]] = {n.id: [] for n in order}
    for n in order:
        for d in n.deps:
            users[d].append(n.id)
    node_lat = {i: sum(e.latency for e in plans[i][0]) for i in plans}
    rank: dict[str, int] = {}
    for n in reversed(order):
        rank[n.id] = node_lat[n.id] + max((rank[u] for u in users[n.id]), default=0)

    lane_free = [0] * lanes
    lane_group = [None] * lanes
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
        ladder, acc = plans[n.id]
        prefix_paid = n.prefix_group is not None and (lane, n.prefix_group) in seen_prefix
        discount = n.prefix_tokens if prefix_paid else 0
        if n.prefix_group is not None:
            seen_prefix.add((lane, n.prefix_group))
        reach, exp_tokens, worst = 1.0, 0.0, 0
        for i, e in enumerate(ladder):
            call = max(0, e.tokens_in - discount) + e.tokens_out
            exp_tokens += reach * call
            worst += call
            reach *= 1.0 - (1.0 if i == len(ladder) - 1 else e.coverage)
        end = start + node_lat[n.id]
        lane_free[lane], lane_group[lane] = end, n.prefix_group
        finish[n.id] = end
        done.add(n.id)
        pending.remove(n)
        total_exp += exp_tokens
        total_worst += worst
        out.append(Assignment(n.id, tuple(e.name for e in ladder), round(acc, 6),
                              round(exp_tokens, 3), worst, lane, start, end))

    if total_worst > token_budget:
        raise CompileError(
            f"worst-case tokens {total_worst} exceed the budget {token_budget}; "
            "lower accuracy bars, add cheaper executors, or raise the budget"
        )
    sched = Schedule(tuple(out), round(total_exp, 3), total_worst, max(finish.values(), default=0))
    executors_used = sorted({name for a in out for name in a.ladder})
    versions = {e.name: e.version for e in executors if e.name in executors_used}
    h = content_hash({
        "assignments": [a.__dict__ for a in sched.assignments],
        "budget": token_budget, "lanes": lanes, "executors": versions,
    })
    return Schedule(sched.assignments, sched.expected_tokens, sched.worst_case_tokens, sched.makespan, h)
