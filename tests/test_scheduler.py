import pytest

from dmc.scheduler import CompileError, Executor, Node, compile_schedule

S1 = Executor("s1.probe", "system1", frozenset({"decide.classify"}), 0, 0, 1,
              {"decide.classify": 0.99}, coverage=0.8)
HUB = Executor("hub.decide", "system2", frozenset({"decide.classify", "analyze.reason"}), 600, 40, 5,
               {"decide.classify": 0.95, "analyze.reason": 0.80})
THINK = Executor("hub.think", "system2", frozenset({"analyze.reason"}), 600, 2000, 40,
                 {"analyze.reason": 0.92})
SPOKE = Executor("spoke.27b", "spoke", frozenset({"analyze.reason", "decide.classify"}), 600, 3000, 90,
                 {"analyze.reason": 0.95, "decide.classify": 0.97})
CODE = Executor("tool.python", "tool", frozenset({"analyze.data"}), 0, 0, 2, {"analyze.data": 1.0})
ALL = [S1, HUB, THINK, SPOKE, CODE]


def test_abstaining_executor_cannot_end_a_ladder_without_forced_accuracy():
    sch = compile_schedule([Node("a", "decide.classify", required_accuracy=0.5)], [S1, HUB], token_budget=10_000)
    assert sch.assignments[0].ladder == ("s1.probe", "hub.decide")
    forced = Executor("s1.forced", "system1", frozenset({"decide.classify"}), 0, 0, 1,
                      {"decide.classify": 0.99}, coverage=0.8, accuracy_all={"decide.classify": 0.9})
    sch = compile_schedule([Node("a", "decide.classify", required_accuracy=0.85)], [forced, HUB], token_budget=10)
    assert sch.assignments[0].ladder == ("s1.forced",) and sch.worst_case_tokens == 0


def test_cheapest_ladder_meeting_the_bar_and_cascade():
    # S1 answers 80% at 0.99; escalating the rest to the hub gives 0.8*0.99 + 0.2*0.95 = 0.982.
    a = compile_schedule([Node("a", "decide.classify", required_accuracy=0.96)], ALL, token_budget=10_000).assignments[0]
    assert a.ladder == ("s1.probe", "hub.decide") and a.expected_accuracy == pytest.approx(0.982)
    assert a.expected_tokens == pytest.approx(0.2 * 640) and a.worst_case_tokens == 640
    # A stricter bar skips the hub and escalates straight to the 27B: 0.792 + 0.2*0.97 = 0.986.
    b = compile_schedule([Node("a", "decide.classify", required_accuracy=0.985)], ALL, token_budget=10_000).assignments[0]
    assert b.ladder == ("s1.probe", "spoke.27b") and b.expected_accuracy == pytest.approx(0.986)


def test_compile_time_errors_are_explicit():
    with pytest.raises(CompileError, match="needs accuracy"):
        compile_schedule([Node("a", "analyze.reason", required_accuracy=0.99)], ALL, token_budget=10**6)
    with pytest.raises(CompileError, match="exceed the budget"):
        compile_schedule([Node("a", "analyze.reason", required_accuracy=0.94)], ALL, token_budget=100)
    with pytest.raises(CompileError, match="cycle"):
        compile_schedule([Node("a", "analyze.data", deps=("b",)), Node("b", "analyze.data", deps=("a",))],
                         ALL, token_budget=10)
    with pytest.raises(CompileError, match="no executor"):
        compile_schedule([Node("a", "create.text")], ALL, token_budget=10)


def test_schedule_is_deterministic_and_respects_dependencies():
    g = [Node("plan", "analyze.reason", required_accuracy=0.9),
         Node("q1", "decide.classify", deps=("plan",), required_accuracy=0.9),
         Node("q2", "decide.classify", deps=("plan",), required_accuracy=0.9),
         Node("sum", "analyze.data", deps=("q1", "q2"))]
    s1 = compile_schedule(g, ALL, token_budget=50_000, lanes=2)
    s2 = compile_schedule(list(reversed(g)), ALL, token_budget=50_000, lanes=2)
    assert s1.hash == s2.hash
    at = {a.node: a for a in s1.assignments}
    assert at["q1"].start >= at["plan"].finish and at["sum"].start >= max(at["q1"].finish, at["q2"].finish)
    assert at["q1"].lane != at["q2"].lane  # independent nodes run in parallel


def test_shared_prefix_is_paid_once_per_lane():
    g = [Node(f"q{i}", "decide.classify", required_accuracy=0.95, prefix_group="doc", prefix_tokens=500)
         for i in range(3)]
    sch = compile_schedule(g, [HUB], token_budget=10_000, lanes=1)
    worst = [a.worst_case_tokens for a in sch.assignments]
    assert worst == [640, 140, 140]
