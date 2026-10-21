from collections import Counter

from dmc.oracle import generate


def test_generation_is_a_pure_function_of_the_seed():
    assert generate(5, seed=3) == generate(5, seed=3)
    assert generate(5, seed=3) != generate(5, seed=4)


def test_balanced_groups_with_consistent_answers():
    items = generate(10, seed=0)
    assert len(items) == 6 * 10 * 3
    assert Counter(i.answer for i in items) == {"yes": 90, "no": 90}
    groups = {}
    for i in items:
        groups.setdefault((i.family, i.group), set()).add(i.answer)
    assert all(len(v) == 1 for v in groups.values())


def test_answers_are_correct_by_construction():
    for i in generate(20, seed=1):
        q = i.question
        if i.family == "sum" and i.paraphrase == 0:
            lhs, rhs = q.removeprefix("Is ").split(" equal to ")
            a, b = map(int, lhs.split(" + "))
            s = int(rhs.split("?")[0])
            assert (a + b == s) == (i.answer == "yes")
        if i.family == "even" and i.paraphrase == 0:
            n = int(q.split()[1])
            assert (n % 2 == 0) == (i.answer == "yes")
