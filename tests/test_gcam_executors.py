"""GCAM v1 executors that run without an LLM: compiled code, classic ML, conformal thresholds."""

import pytest

from gcam.domain import build_world, claims, estimates, reply_pairs, tool_requests
from gcam.executors.classic import ExpenseGBDT, OrderForecaster, PolicyTfidf, conformal_threshold
from gcam.executors.code import (CLAIM_RULES, EffectLog, TOOL_RULES, arith, build_code_executors, keyword_classifier,
                                 refund_rule, reply_checks, safe_eval, summary_checks)
from gcam.domain import NOW


@pytest.fixture(scope="module")
def world():
    return build_world()


def test_refund_rule_follows_the_policy_sections():
    base = {"status": "delivered", "order_date": "2026-09-01", "delivered_date": "2026-09-05", "category": "home"}
    assert refund_rule(base, "damaged", NOW)["decision"] == "approve"  # 46 days, limit 90
    assert refund_rule(base, "changed_mind", NOW)["decision"] == "deny"  # limit 30
    assert refund_rule({**base, "category": "grocery", "delivered_date": "2026-10-15"}, "changed_mind", NOW)["rule"] == "P1"
    lost = {"status": "lost_in_transit", "order_date": "2026-10-01", "delivered_date": None, "category": "toys"}
    assert refund_rule(lost, "not_received", NOW)["decision"] == "approve"  # 20 days without delivery
    assert refund_rule({**lost, "order_date": "2026-10-18"}, "not_received", NOW)["decision"] == "deny"  # wait


def test_tool_args_resolve_dates_and_report_missing_slots(world):
    ex = build_code_executors(world, EffectLog())
    out = ex["code.args"].run("tool.args", {"tool": "schedule_callback", "text": "Please call me tomorrow at 3pm.",
                                            "customer": "C-1001", "now": NOW}, forced=False).value
    assert out["args"] == {"customer_id": "C-1001", "date": "2026-10-22", "time": "15:00"} and not out["missing"]
    weekend = ex["code.args"].run("tool.args", {"tool": "schedule_callback", "text": "Call me on Saturday at 3pm.",
                                                "customer": "C-1001", "now": NOW}, forced=False).value
    assert weekend["invalid"]  # P6: weekdays only
    missing = ex["code.args"].run("tool.args", {"tool": "cancel_order", "text": "Cancel my order please.",
                                                "customer": "C-1001", "now": NOW}, forced=False).value
    assert missing["missing"] == ["order_id"]


def test_destructive_tools_need_explicit_approval_and_effects_are_idempotent(world):
    log = EffectLog()
    ex = build_code_executors(world, log)
    args = {"tool": "cancel_order", "args": {"order_id": "A-5000"}, "missing": [], "invalid": []}
    refused = ex["code.confirm"].run("meta.confirm", {"tool": "cancel_order", "args": args, "autonomy": "confirm",
                                                      "approvals": []}, forced=False).value
    granted = ex["code.confirm"].run("meta.confirm", {"tool": "cancel_order", "args": args, "autonomy": "confirm",
                                                      "approvals": ["cancel_order:A-5000"]}, forced=False).value
    assert not refused["approved"] and granted["approved"]
    payload = {"tool": "schedule_callback", "args": {"args": {"customer_id": "C-1001", "date": "2026-10-22",
                                                               "time": "15:00"}}, "approved": granted}
    a = ex["code.tools"].run("tool.execute", payload, forced=False).value
    b = ex["code.tools"].run("tool.execute", payload, forced=False).value
    assert a == b and len(log.effects) == 1  # replay never applies a side effect twice (R22)


def test_reply_verifier_catches_each_defect_kind(world):
    for pair in reply_pairs(10, seed=3, world=world):
        name = world.customer(pair.customer)["name"]
        good, bad = (pair.reply_a, pair.reply_b) if pair.better == "A" else (pair.reply_b, pair.reply_a)
        assert all(reply_checks(good, name, pair.order_id).values())
        caught = not all(reply_checks(bad, name, pair.order_id).values())
        # Tone defects are invisible to code; those pairs go to a judge.
        assert caught or pair.defect in ("rude",)


def test_keyword_rules_cover_canonical_phrasings_and_abstain_elsewhere(world):
    kw = keyword_classifier(CLAIM_RULES)
    train = claims(80, split="train", seed=2)
    decided = [kw({"text": c.text}, False) for c in train]
    acc = sum(r.value == c.reason for r, c in zip(decided, train) if r.answered) / sum(r.answered for r in decided)
    assert acc >= 0.95 and sum(r.answered for r in decided) >= 60
    tools = keyword_classifier(TOOL_RULES)
    req = tool_requests(60, split="train", seed=2, order_ids=["A-5000"])
    dec = [tools({"text": r.text}, False) for r in req]
    assert sum(d.value == r.tool for d, r in zip(dec, req) if d.answered) / max(1, sum(d.answered for d in dec)) >= 0.9


def test_arith_grammar_and_safe_eval():
    train = estimates(10, split="train", seed=4)
    for e in train:
        assert arith({"text": e.text}, False).value == e.answer
    assert not arith({"text": estimates(2, split="test", seed=4)[0].text}, False).answered
    assert safe_eval("0.12 * 340 * 45") == pytest.approx(1836.0)
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('true')")


def test_analytics_finds_the_planted_top_category(world):
    ex = build_code_executors(world, EffectLog())
    facts = ex["code.analytics"].run("analyze.refund_rates", {"days": 180, "now": NOW}, forced=False).value
    assert facts["top_category"] == "apparel" and facts["significant"]
    text = ex["code.analytics"].run("create.summary_template", {"facts": facts}, forced=False).value
    assert all(summary_checks(text, facts).values())


def test_conformal_threshold_meets_its_target():
    conf = [0.95, 0.9, 0.8, 0.7, 0.6, 0.55]
    correct = [True, True, True, False, True, False]
    tau = conformal_threshold(conf, correct, 0.99)
    assert tau == 0.8
    assert conformal_threshold([0.5, 0.6], [False, False], 0.9) == float("inf")


def test_gbdt_abstains_in_the_band_where_the_memo_decides(world):
    rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in world.expenses.to_dict("records")]
    gbdt = ExpenseGBDT().fit(rows[:1000])
    tau = gbdt.calibrate(rows[1000:1200], target=0.97)
    out = [gbdt.run("decide.expense_review", {"row": r}, forced=False) for r in rows[1200:1400]]
    answered = [o for o in out if o.answered]
    acc = sum(o.value["needs_review"] == r["needs_review"] for o, r in zip(out, rows[1200:1400]) if o.answered)
    assert tau < 1 and 0.4 < len(answered) / len(out) < 1.0
    assert acc / len(answered) >= 0.9


def test_forecaster_beats_or_matches_seasonal_naive(world):
    fc = OrderForecaster(world.daily_orders).run("forecast.orders", {"days": 7, "now": NOW}, forced=False).value
    assert len(fc["forecast"]) == 7 and fc["backtest_mae"] <= fc["baseline_mae"]
    assert all(r["lo"] <= r["point"] <= r["hi"] for r in fc["forecast"])


def test_tfidf_retrieves_the_right_section():
    r = PolicyTfidf()
    assert r.run("retrieve.policy", {"query": "damaged items"}, forced=True).value["id"] == "P2"
    assert r.run("retrieve.policy", {"query": "callbacks on weekdays"}, forced=True).value["id"] == "P6"


class _FakeLM:
    """Stands in for a model: candidate text depends only on the sampling seed."""

    role = "hub"
    model_id = "fake"

    def __init__(self):
        self.seeds = []

    def render(self, system, user):
        return system + "\n" + user

    def n_tokens(self, text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens, *, seed=None):
        self.seeds.append(seed)
        if seed is not None and seed % 2:  # odd seeds write a reply that passes every check
            text = ("Hi Ana, thank you for telling us about order A-1. We will refund you and you will hear "
                    "from us within 24 hours.")
        else:
            text = "Sorry about that."
        return text, list(range(len(text.split()))), -1.0 - (seed or 0) % 7 / 10


def test_best_of_k_reply_is_seeded_by_content_and_selects_a_verified_candidate():
    from gcam.executors.llm import LLMExecutor, _seed

    p = {"customer_name": "Ana Lopez", "order_id": "A-1", "product": "lamp", "reason": "damaged",
         "decision": "approve", "rule": "P2"}
    lm = _FakeLM()
    res = LLMExecutor("hub", lm, k=3).run("create.reply", p, forced=False)
    assert lm.seeds == [_seed(p, i) for i in range(3)] and len(set(lm.seeds)) == 3
    again = _FakeLM()
    assert LLMExecutor("hub", again, k=3).run("create.reply", p, forced=False).value == res.value  # replayable
    passing = [c["passed"] for c in res.detail["candidates"]]
    assert passing == [s % 2 == 1 for s in lm.seeds]  # the verifier ran on every candidate
    assert res.answered == any(passing) and (not any(passing) or "A-1" in res.value)
    forced = LLMExecutor("hub", _FakeLM(), k=1).run("create.reply_fallback", p, forced=True)
    assert forced.answered and forced.detail["candidates"][0]["passed"] is False  # k = 1 is greedy (seed None)


def test_reply_verifier_reads_denials_and_invented_offers():
    deny = "Hi Kofi, we are unable to issue a refund for order A-1 under policy P3. You will hear from us within 24 hours."
    assert all(reply_checks(deny, "Kofi Khan", "A-1", "deny").values())
    promised = "Hi Kofi, we have refunded order A-1 in full. You will hear from us within 24 hours."
    assert not reply_checks(promised, "Kofi Khan", "A-1", "deny")["no_refund_promise"]
    offer = "Hi Kofi, sorry about order A-1. We will replace it and add store credit. We'll be in touch."
    assert not reply_checks(offer, "Kofi Khan", "A-1", "deny")["no_false_promise"]
    approve = "Hi Kofi, your refund for order A-1 has been approved under policy P2. We'll confirm by email."
    assert all(reply_checks(approve, "Kofi Khan", "A-1", "approve").values())


def test_held_out_period_memos_use_unseen_wording(world):
    from datetime import timedelta

    from gcam.domain import (HELD_OUT_DAYS, MEMO_CLEAR, MEMO_CLEAR_TEST, MEMO_RISK_DOWN, MEMO_RISK_DOWN_TEST,
                             MEMO_RISK_UP, MEMO_RISK_UP_TEST, NOW_DT)

    def stems(memos):  # template text before any placeholder
        return tuple(m.split("{")[0] for m in memos)

    train = stems((*MEMO_CLEAR.values(), *MEMO_RISK_UP, *MEMO_RISK_DOWN))
    test = stems((*MEMO_CLEAR_TEST.values(), *MEMO_RISK_UP_TEST, *MEMO_RISK_DOWN_TEST))
    cut = (NOW_DT - timedelta(days=HELD_OUT_DAYS)).date().isoformat()
    ex = world.expenses
    assert all(m.startswith(test) and not m.startswith(train) for m in ex[ex.date >= cut].memo)
    assert all(m.startswith(train) for m in ex[ex.date < cut].memo)


def test_memo_signal_decides_borderline_reports_and_defers_on_neutral_memos():
    import numpy as np

    from gcam.executors.system1 import EncoderHeads, LinearHead

    class BagEncoder:  # deterministic stand-in for MiniLM: hashed bag of words
        def embed(self, text):
            v = np.zeros(64)
            for w in text.lower().split():
                v[sum(map(ord, w)) % 64] += 1.0
            return v / (np.linalg.norm(v) or 1.0), len(text.split())

    class StubGBDT:  # the structured-field model: says "review" for amounts over 500
        def proba(self, rows):
            return np.array([0.9 if r["amount"] > 500 else 0.1 for r in rows])

    enc = BagEncoder()
    memos = [("receipt lost, amount from memory", "review"), ("cash payment, no invoice available", "review"),
             ("pre-approved by finance, approval ref FIN-1234", "approve"),
             ("client-billable, recharged on invoice INV-5678", "approve"),
             ("printer paper and toner for the office", "neither"), ("train to client site for workshop", "neither")] * 5
    head = LinearHead(("approve", "neither", "review")).fit(np.stack([enc.embed(m)[0] for m, _ in memos]),
                                                            [y for _, y in memos])
    head.tau = 0.0
    s1 = EncoderHeads(enc, {"memo.signal": head}, gbdt=StubGBDT())
    row = {"expense_id": "E-1", "memo": "receipt lost, amount from memory", "amount": 20.0}
    res = s1.run("decide.expense_review", {"row": row}, forced=False)
    assert res.answered and res.value == {"expense_id": "E-1", "needs_review": True}  # the memo decides
    assert res.usage == {"encoder": 5} and res.detail["signal"] == "review"
    neutral = {"expense_id": "E-2", "memo": "printer paper and toner for the office", "amount": 900.0}
    res = s1.run("decide.expense_review", {"row": neutral}, forced=False)
    assert res.detail["signal"] == "neither" and res.value["needs_review"] is True  # deferred to the GBDT


def test_agreement_answers_only_when_both_views_agree():
    from gcam.executors.base import ExecResult
    from gcam.executors.system1 import Agreement

    class View:
        def __init__(self, label, conf, usage):
            self.heads = {"tool.select": None}
            self.label, self.conf, self.usage = label, conf, usage

        def run(self, op, payload, *, forced):
            return ExecResult(True, self.label, self.conf, self.usage)

    both = Agreement(View("cancel_order", 0.9, {"encoder": 5}), View("cancel_order", 0.8, {"hub": 7}), {"tool.select": 0.7})
    r = both.run("tool.select", {"text": "x"}, forced=False)
    assert r.answered and r.value == "cancel_order" and r.confidence == 0.8 and r.usage == {"encoder": 5, "hub": 7}
    split = Agreement(View("get_order_status", 0.95, {"encoder": 5}), View("cancel_order", 0.6, {"hub": 7}),
                      {"tool.select": 0.0})
    assert not split.run("tool.select", {"text": "x"}, forced=False).answered  # disagreement escalates
    assert split.run("tool.select", {"text": "x"}, forced=True).value == "get_order_status"  # forced: the surer view
