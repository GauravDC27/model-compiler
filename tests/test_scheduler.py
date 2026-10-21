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
CODE = Executor("tool.python", "tool", frozenset({"analyze.data"}), 0, 0, 2, {"analyze.data": 1.0},
                input_reads=0)
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


def test_nodes_requiring_bitwise_determinism_only_use_d1_executors():
    d1_s1 = Executor("s1.head", "system1", frozenset({"decide.classify"}), 0, 0, 1,
                     {"decide.classify": 0.99}, coverage=0.7, contract="D1")
    d1_dense = Executor("dense.student", "system2", frozenset({"decide.classify"}), 300, 0, 3,
                        {"decide.classify": 0.93}, contract="D1")
    hub_d0 = Executor("hub.decide", "system2", frozenset({"decide.classify"}), 200, 0, 2,
                      {"decide.classify": 0.97}, contract="D0")
    loose = compile_schedule([Node("a", "decide.classify", required_accuracy=0.95)],
                             [d1_s1, d1_dense, hub_d0], token_budget=10_000).assignments[0]
    strict = compile_schedule([Node("a", "decide.classify", required_accuracy=0.95, requires_contract="D1")],
                              [d1_s1, d1_dense, hub_d0], token_budget=10_000).assignments[0]
    assert "hub.decide" in loose.ladder
    assert strict.ladder == ("s1.head", "dense.student")
    with pytest.raises(CompileError, match="contract D1"):
        compile_schedule([Node("a", "analyze.reason", requires_contract="D1")], [hub_d0, THINK], token_budget=10**6)


def test_joint_calibration_accounts_for_correlated_errors():
    from dmc.scheduler import Outcome

    # S1 answers half the items correctly; the other half (hard) it abstains on. The hub is 90%
    # accurate marginally but only 80% on the hard half: independence would over-predict.
    s1 = Executor("s1", "system1", frozenset({"op"}), 0, 0, 1, {"op": 1.0}, coverage=0.5)
    hub = Executor("hub", "system2", frozenset({"op"}), 100, 0, 1, {"op": 0.9})
    items = []
    for i in range(10):
        easy = i < 5
        items.append({
            "s1": Outcome(answered=easy, correct=easy, correct_forced=easy, tokens=0),
            # Hard items are also longer: the hub spends 200 tokens on them, not its mean of 100.
            "hub": Outcome(answered=True, correct=easy or i < 9, correct_forced=easy or i < 9,
                           tokens=100 if easy else 200),
        })
    node = Node("a", "op", required_accuracy=0.85)
    ind = compile_schedule([node], [s1, hub], token_budget=10**6).assignments[0]
    joint = compile_schedule([node], [s1, hub], token_budget=10**6, calibration={"op": items}).assignments[0]
    assert ind.estimate == "independent" and joint.estimate == "joint"
    assert ind.ladder == joint.ladder == ("s1", "hub")
    assert ind.expected_accuracy == pytest.approx(0.95)  # 0.5*1.0 + 0.5*0.9
    assert joint.expected_accuracy == pytest.approx(0.9)  # 5 easy + 4 of 5 hard
    assert ind.expected_tokens == pytest.approx(50)  # half escalate, at the hub's mean cost
    assert joint.expected_tokens == pytest.approx(100)  # half escalate, at what they really cost
    with pytest.raises(CompileError):
        compile_schedule([Node("a", "op", required_accuracy=0.93)], [s1, hub], token_budget=10**6,
                         calibration={"op": items})


def test_tool_output_cap_counts_against_worst_case_and_lock_hash_changes_schedule():
    a = compile_schedule([Node("a", "decide.classify", required_accuracy=0.9, input_cap=1000)], [HUB],
                         token_budget=10**6)
    assert a.worst_case_tokens == 600 + 1000 + 40
    b = compile_schedule([Node("a", "decide.classify", required_accuracy=0.9, input_cap=1000)], [HUB],
                         token_budget=10**6, lock_hash="sha256:other")
    assert a.hash != b.hash


def test_prefix_discount_does_not_cross_executors():
    g = [Node(f"q{i}", "decide.classify", required_accuracy=0.96, prefix_group="doc", prefix_tokens=500)
         for i in range(2)]
    sch = compile_schedule(g, [S1, HUB], token_budget=10**6, lanes=1)
    # Ladder s1 -> hub: s1 consumes no prompt, so only the hub rung pays the prefix, once.
    worst = [a.worst_case_tokens for a in sch.assignments]
    assert worst == [640, 140]


def test_program_bounds_guards_variants_and_capabilities():
    from dmc.scheduler import Guard, GuardRuntime, Program, compile_program

    main = (Node("run", "analyze.data"), Node("check", "decide.classify", deps=("run",), required_accuracy=0.9))
    fix = (Node("repair", "analyze.reason", required_accuracy=0.9),)
    prog = Program(main, {"fix": fix}, (Guard("g", "check", "tests_pass", "fix"),
                                        Guard("g2", "check", "nonempty_result", "recompile")),
                   recompile_allowance=1, recompile_budget=500)
    ps = compile_program(prog, ALL, token_budget=10**6)
    assert ps.worst_case_tokens == ps.main.worst_case_tokens + ps.variants["fix"].worst_case_tokens + 500
    rt = GuardRuntime(prog)
    assert rt.on_failure("g") == "fix" and rt.on_failure("g") == "abstain"  # no silent loops
    assert rt.on_failure("g2") == "recompile" and rt.on_failure("g2") == "abstain"
    with pytest.raises(CompileError, match="exceed the budget"):
        compile_program(prog, ALL, token_budget=ps.worst_case_tokens - 1)
    risky = Program((Node("send", "analyze.data", capabilities=frozenset({"external_send"})),))
    with pytest.raises(CompileError, match="beyond the ceiling"):
        compile_program(risky, ALL, token_budget=10**6)
    with pytest.raises(CompileError, match="allowance is 0"):
        compile_program(Program(main, guards=(Guard("g", "check", "p", "recompile"),)), ALL, token_budget=10**6)


def test_input_reads_scale_the_capped_input():
    # Code never feeds the input to a model; order-averaged scoring reads it once per order.
    assert CODE.call_tokens(1000) == 0
    two_orders = Executor("hub.scored", "system2", frozenset({"decide.classify"}), 600, 0, 5,
                          {"decide.classify": 0.97}, input_reads=2)
    assert two_orders.call_tokens(1000) == 600 + 2 * 1000
    assert two_orders.call_tokens(1000, discount=500) == 600 + 2 * 1000 - 500
