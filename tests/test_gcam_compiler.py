"""GCAM v1 front end: domain determinism, grammar parser, Goal IR, template planner."""

import pytest

from dmc.scheduler import CompileError, compile_program
from gcam.domain import build_world, claims, reply_pairs
from gcam.goal_ir import Clarify, Goal
from gcam.parser import parse
from gcam.planner import Planner, expense_rows


@pytest.fixture(scope="module")
def world():
    return build_world()


def _order_of(world):
    row = world.orders[world.orders.status == "delivered"].iloc[3]
    return row.customer_id, row.order_id


def test_world_is_a_pure_function_of_the_seed(world):
    assert build_world().snapshot_hash() == world.snapshot_hash()
    assert build_world(seed=8).snapshot_hash() != world.snapshot_hash()


def test_train_and_held_out_phrasings_do_not_overlap():
    train = {c.text for c in claims(200, split="train", seed=1)}
    test = {c.text for c in claims(200, split="test", seed=1)}
    assert not train & test


def test_parse_every_supported_form(world):
    cust, order = _order_of(world)
    pair = reply_pairs(1, seed=0, world=world)[0]
    cases = {
        f'Customer {cust} requests a refund for order {order}: "The lamp arrived broken."': "refund_decision",
        f'Handle request from {cust}: "Where is my order {order}?"': "support_request",
        f'Which reply to {pair.customer} about order {pair.order_id} is better? A: "{pair.reply_a}" B: "{pair.reply_b}"':
            "compare_replies",
        f"Write a reply to {cust} about order {order} that arrived damaged": "write_reply",
        "Which category had the highest refund rate in the last 90 days?": "refund_analytics",
        "Estimate: If we refund 10% of 500 orders that average $40, what is the total refund cost?": "cost_estimate",
        "Review expense reports from the last 7 days": "expense_review",
        "Forecast daily orders for the next 7 days": "forecast_orders",
        "What does our policy say about damaged items?": "policy_question",
    }
    for text, family in cases.items():
        goal = parse(text, world)
        assert isinstance(goal, Goal), (text, goal)
        assert goal.family == family
    assert parse(f"Write a reply to {cust} about order {order} that arrived damaged", world).slots["reason"] == "damaged"


def test_context_minimization(world):
    cust, order = _order_of(world)
    timed = parse(f'Customer {cust} requests a refund for order {order}: "It broke."', world)
    untimed = parse("Estimate: What does it cost to refund 5 percent of 120 orders at $12 each?", world)
    assert timed.context.now is not None and timed.context.session_state_hash == world.snapshot_hash()
    assert untimed.context.now is None and untimed.context.session_state_hash is None


def test_clarify_instead_of_guessing(world):
    cust, order = _order_of(world)
    missing = parse(f'Customer {cust} requests a refund: "It broke."', world)
    assert isinstance(missing, Clarify) and missing.candidates
    other = world.customers.customer_id[(world.customers.customer_id != cust).idxmax()]
    wrong_owner = parse(f'Customer {other} requests a refund for order {order}: "It broke."', world)
    assert isinstance(wrong_owner, Clarify) and "another customer" in wrong_owner.question
    assert isinstance(parse("Make me rich", world), Clarify)
    assert isinstance(parse("Forecast daily orders for the next 90 days", world), Clarify)


def test_plans_have_the_expected_shape_and_are_cached(world):
    cust, order = _order_of(world)
    planner = Planner()
    g1 = parse(f'Customer {cust} requests a refund for order {order}: "It broke."', world)
    g2 = parse(f'Customer {cust} requests a refund for order {order}: "Never came."', world)
    p1, p2 = planner.plan(g1, world), planner.plan(g2, world)
    assert p1 is p2 and planner.hits == 1  # same template, slots substituted at run time
    assert [n.op for n in p1.program.main] == ["tool.get_order", "decide.claim_reason", "decide.refund"]
    write = planner.plan(parse(f"Write a reply to {cust} about order {order} that arrived damaged", world), world)
    assert set(write.program.variants) == {"spoke_reply"}
    exp_goal = parse("Review expense reports from the last 7 days", world)
    exp = planner.plan(exp_goal, world)
    assert len(exp.program.main) == len(expense_rows(exp_goal, world)) + 1


def test_autonomy_sets_the_capability_ceiling(world):
    cust, _ = _order_of(world)
    text = f'Handle request from {cust}: "Cancel order A-5001."'
    read_only = parse(text, world, autonomy="read_only")
    plan = Planner().plan(read_only, world)
    with pytest.raises(CompileError, match="beyond the ceiling"):
        compile_program(plan.program, [], token_budget=10**6)
