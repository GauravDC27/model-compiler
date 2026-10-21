"""LLM executors: the hub (with goal-family capacity) and the specialist spoke.

Decision-bearing values are extracted by scoring single-token option letters in one forward
pass per option ordering, averaged over the canonical and the reversed ordering (R17). Nothing
decision-bearing is read from sampled text. Generation is used only where the output is text:

* ``create.reply``: best-of-K seeded candidates. Every candidate is run through the executable
  verifier, and the passing candidate the model itself finds most likely wins. This is
  selection, never fusion (plan §I.2.4).
* ``create.summary``: one greedy draft checked against the computed facts.
* ``analyze.estimate``: program-of-thought. The model writes one arithmetic expression, which
  is parsed and evaluated by code; an expression that does not parse is an abstention.
"""

from __future__ import annotations

import math
import re

import numpy as np

from dmc.canonical import content_hash

from ..domain import POLICY
from ..models import LETTERS, LM
from .base import ExecResult, NotSupported
from .code import reply_checks, safe_eval, summary_checks

CLAIM_OPTIONS = {
    "damaged": "the item arrived damaged, broken or defective",
    "not_received": "the item never arrived or was not delivered",
    "wrong_item": "the wrong item, size or colour was sent",
    "changed_mind": "the item is fine but the customer no longer wants it",
}
TOOL_OPTIONS = {
    "get_order_status": "look up where an order is or when it will arrive",
    "schedule_callback": "book a phone call with the customer at a given day and time",
    "cancel_order": "cancel an order so it is not shipped",
    "update_address": "change the customer's shipping address",
    "search_policy": "answer a question about our rules and policies",
    "open_refund_case": "start a refund for a specific order",
}
REVIEW_OPTIONS = {"approve": "approve automatically", "review": "needs manual review"}

SYSTEMS = {
    "decide.claim_reason": "You classify customer refund claims. Reply with the letter of the best option only.",
    "tool.select": "You route customer requests to the one tool that handles them. Reply with the letter of the tool only.",
    "decide.expense_review": "You review company expense reports against the expense policy. Reply with the letter only.",
    "retrieve.policy": "You find the policy section that answers a question. Reply with the letter of the section only.",
    "evaluate.pairwise": ("You compare two customer-service replies. The better reply is accurate, polite, addresses the "
                          "right customer and order, promises nothing extra, and says what happens next. Reply 1 or 2 only."),
    "create.reply": ("You write short, warm, accurate customer-service replies. Use the customer's first name, mention "
                     "the order number, say what happens next, and never promise anything that is not stated."),
    "create.summary": ("You write a two-sentence summary of an analysis for a manager. Use only the numbers given, "
                       "exactly as written, and say whether the difference is statistically significant."),
    "analyze.estimate": ("You turn a business arithmetic question into a single Python arithmetic expression. Reply with the "
                         "expression only: numbers, + - * / and parentheses, no words and no units."),
}


def _options_text(labels: tuple[str, ...], desc: dict[str, str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {desc[lab]}" for i, lab in enumerate(labels))


def claim_user(payload: dict, order: tuple[str, ...]) -> str:
    return f"Claim: {payload['text']}\n\nWhat is the reason for this claim?\n{_options_text(order, CLAIM_OPTIONS)}"


def tool_user(payload: dict, order: tuple[str, ...]) -> str:
    return f"Request: {payload['text']}\n\nWhich tool handles this request?\n{_options_text(order, TOOL_OPTIONS)}"


def expense_user(payload: dict, order: tuple[str, ...]) -> str:
    r = payload["row"]
    return (f"Policy: {POLICY[4].text}\n\nExpense report {r['expense_id']}: ${r['amount']:.2f} for "
            f"{r['category']} at {r['vendor']}; receipt {'attached' if r['has_receipt'] else 'missing'}; "
            f"submitted on a {'weekend' if r['weekend'] else 'weekday'}; submitter tenure {r['tenure_months']} "
            f"months; {'possible duplicate within 7 days' if r['duplicate_7d'] else 'no duplicate'}.\n"
            f"Memo: {r['memo']}\n\nDecision?\n{_options_text(order, REVIEW_OPTIONS)}")


POLICY_DESC = {s.id: f"{s.title}: {s.text}" for s in POLICY}


def policy_user(payload: dict, order: tuple[str, ...]) -> str:
    return f"Question: {payload['query']}\n\nWhich section answers it?\n{_options_text(order, POLICY_DESC)}"


def pair_user(p: dict, first: str, second: str) -> str:
    return (f"Customer: {p['customer_name']}, order {p['order_id']}.\n\nReply 1:\n{p[first.lower()]}\n\n"
            f"Reply 2:\n{p[second.lower()]}\n\nWhich reply is better? Answer 1 or 2.")


def reply_user(p: dict) -> str:
    first = p["customer_name"].split()[0]
    issue = {"damaged": "the item arrived damaged", "not_received": "the order has not arrived",
             "wrong_item": "the wrong item was delivered", "changed_mind": "the customer changed their mind"}[p["reason"]]
    decision = (f"refund approved under policy {p['rule']}" if p["decision"] == "approve"
                else f"no refund under policy {p['rule']}; explain kindly and offer the next step")
    return (f"Customer first name: {first}\nOrder: {p['order_id']} ({p['product']})\nIssue: {issue}\n"
            f"Decision: {decision}\nWrite the reply in 2 to 4 sentences.")


def summary_user(p: dict) -> str:
    f = p["facts"]
    return (f"Window: last {f['days']} days. Category with the highest refund rate: {f['top_category']}, "
            f"{100 * f['top_rate']:.1f}%. All other categories together: {100 * f['rest_rate']:.1f}%. "
            f"Two-proportion z-test: z = {f['z']}, p = {f['p_value']}, "
            f"{'statistically significant' if f['significant'] else 'not statistically significant'} at the 5% level.\n"
            "Write the two-sentence summary.")


CHOICE = {
    "decide.claim_reason": (CLAIM_OPTIONS, claim_user),
    "tool.select": (TOOL_OPTIONS, tool_user),
    "decide.expense_review": (REVIEW_OPTIONS, expense_user),
    "retrieve.policy": (POLICY_DESC, policy_user),
}
# Payload fields that carry the node's capped input (truncated by the runtime to the node's cap).
INPUT_FIELDS = {"decide.claim_reason": ("text",), "tool.select": ("text",), "decide.expense_review": ("row.memo",),
                "retrieve.policy": ("query",), "evaluate.pairwise": ("a", "b"), "analyze.estimate": ("text",)}


def prompts(op: str, payload: dict, k: int = 3) -> list[tuple[str, str]]:
    """The (system, user) prompts an executor processes for one call, in order."""
    if op in CHOICE:
        canon = tuple(sorted(CHOICE[op][0]))
        return [(SYSTEMS[op], CHOICE[op][1](payload, o)) for o in (canon, tuple(reversed(canon)))]
    if op == "evaluate.pairwise":
        return [(SYSTEMS[op], pair_user(payload, a, b)) for a, b in (("A", "B"), ("B", "A"))]
    if op in ("create.reply", "create.reply_fallback"):
        return [(SYSTEMS["create.reply"], reply_user(payload))] * (k if op == "create.reply" else 1)
    if op == "create.summary":
        return [(SYSTEMS[op], summary_user(payload))]
    if op == "analyze.estimate":
        return [(SYSTEMS[op], payload["text"])]
    raise NotSupported(op)


def blank_inputs(op: str, payload: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in payload.items()}
    for field in INPUT_FIELDS.get(op, ()):
        if "." in field:
            a, b = field.split(".")
            out[a][b] = ""
        else:
            out[field] = ""
    return out


class LLMExecutor:
    def __init__(self, name: str, lm: LM, *, family: dict[str, str | None] | None = None, k: int = 3,
                 max_reply_tokens: int = 70):
        self.name = name
        self.lm = lm
        self.family = family or {}  # op -> goal family whose capacity is switched on
        self.k = k
        self.max_reply_tokens = max_reply_tokens
        self.tau: dict[str, float] = {}  # op -> calibrated confidence threshold
        self.version = f"{name}/1;{lm.model_id}"

    # ---------------------------------------------------------------- multiple choice
    def _choice(self, op: str, labels: tuple[str, ...], user_fn) -> tuple[str, float, float, int, dict]:
        """Order-averaged option scoring: (label, confidence, margin, prompt tokens, scores)."""
        canon = tuple(sorted(labels))
        orders = [canon, tuple(reversed(canon))]
        totals = {lab: 0.0 for lab in canon}
        tokens = 0
        with self.lm.family(self.family.get(op)):
            for order in orders:
                prompt = self.lm.render(SYSTEMS[op], user_fn(order))
                lp, n = self.lm.choice_logprobs(prompt, LETTERS[: len(order)])
                tokens += n
                for lab, v in zip(order, lp):
                    totals[lab] += float(v) / len(orders)
        ranked = sorted(totals, key=lambda lab: (-totals[lab], lab))
        z = np.array([totals[lab] for lab in ranked])
        p = np.exp(z - z.max())
        p /= p.sum()
        return ranked[0], float(p[0]), float(z[0] - z[1]), tokens, {k: round(v, 4) for k, v in totals.items()}

    def _decided(self, op: str, conf: float, forced: bool) -> bool:
        return forced or conf >= self.tau.get(op, 0.0)

    def template_tokens(self, op: str, sample: dict) -> int:
        """Prompt tokens per call with the capped input fields blanked: the fixed part of the bound."""
        return sum(self.lm.n_tokens(self.lm.render(sys_, user)) for sys_, user in prompts(op, blank_inputs(op, sample), self.k))

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        role = self.lm.role
        if op in CHOICE:
            options, user_fn = CHOICE[op]
            lab, conf, margin, n, sc = self._choice(op, tuple(options), lambda order: user_fn(payload, order))
            value = lab
            if op == "decide.expense_review":
                value = {"expense_id": payload["row"]["expense_id"], "needs_review": lab == "review"}
            elif op == "retrieve.policy":
                sec = next(s for s in POLICY if s.id == lab)
                value = {"id": sec.id, "title": sec.title, "score": round(conf, 4)}
            return ExecResult(self._decided(op, conf, forced), value, conf, {role: n}, {"margin": margin, "scores": sc})
        if op == "evaluate.pairwise":
            return self._pairwise(payload, forced)
        if op in ("create.reply", "create.reply_fallback"):
            return self._reply(payload, forced, k=1 if op == "create.reply_fallback" else self.k)
        if op == "create.summary":
            return self._summary(payload, forced)
        if op == "analyze.estimate":
            return self._estimate(payload, forced)
        raise NotSupported(f"{self.name} cannot run {op}")

    def _pairwise(self, p: dict, forced: bool) -> ExecResult:
        op = "evaluate.pairwise"
        scores = {"A": 0.0, "B": 0.0}
        tokens = 0
        for first, second in (("A", "B"), ("B", "A")):  # both orders: position bias cancels (Judge MUST)
            lp, n = self.lm.choice_logprobs(self.lm.render(SYSTEMS[op], pair_user(p, first, second)), "12")
            tokens += n
            scores[first] += float(lp[0]) / 2
            scores[second] += float(lp[1]) / 2
        better = "A" if scores["A"] >= scores["B"] else "B"
        margin = abs(scores["A"] - scores["B"])
        conf = 1 / (1 + math.exp(-margin))
        return ExecResult(self._decided(op, conf, forced), better, conf, {self.lm.role: tokens},
                          {"margin": margin, "scores": scores})

    def _reply(self, p: dict, forced: bool, k: int) -> ExecResult:
        user = reply_user(p)
        prompt = self.lm.render(SYSTEMS["create.reply"], user)
        n_prompt = self.lm.n_tokens(prompt)
        cands, used = [], {self.lm.role: 0}
        for i in range(k):
            seed = None if k == 1 else _seed(p, i)
            text, ids, mean_lp = self.lm.generate(prompt, self.max_reply_tokens, seed=seed)
            used[self.lm.role] += n_prompt + len(ids)
            checks = reply_checks(text, p["customer_name"], p["order_id"], p["decision"])
            cands.append({"text": text, "mean_logprob": round(mean_lp, 4), "checks": checks, "passed": all(checks.values())})
        passing = [c for c in cands if c["passed"]]
        pool = passing or cands
        best = sorted(range(len(pool)), key=lambda j: (-pool[j]["mean_logprob"], j))[0]
        return ExecResult(bool(passing) or forced, pool[best]["text"], len(passing) / k, used,
                          {"candidates": cands, "selected": best, "verified": bool(passing)})

    def _summary(self, p: dict, forced: bool) -> ExecResult:
        f = p["facts"]
        user = summary_user(p)
        prompt = self.lm.render(SYSTEMS["create.summary"], user)
        text, ids, _ = self.lm.generate(prompt, 90)
        checks = summary_checks(text, f)
        ok = all(checks.values())
        return ExecResult(ok or forced, text, float(ok), {self.lm.role: self.lm.n_tokens(prompt) + len(ids)},
                          {"checks": checks, "verified": ok})

    def _estimate(self, p: dict, forced: bool) -> ExecResult:
        prompt = self.lm.render(SYSTEMS["analyze.estimate"], p["text"])
        text, ids, _ = self.lm.generate(prompt, 48)
        expr = text.strip().strip("`").replace("python", "").strip().splitlines()[0] if text.strip() else ""
        expr = re.sub(r"^[A-Za-z_ ]+=", "", expr).strip()
        usage = {self.lm.role: self.lm.n_tokens(prompt) + len(ids)}
        try:
            value = round(safe_eval(expr), 2)
            return ExecResult(True, value, 1.0, usage, {"expression": expr})
        except Exception as e:  # unparseable program: abstain (or a defined wrong answer when forced)
            return ExecResult(forced, None, 0.0, usage, {"expression": expr, "error": str(e)[:80]})
