import pytest

from dmc import metrics as m


def test_tar_and_unique():
    runs = [["a", "a", "a"], ["a", "b", "a"]]
    assert m.tar_raw(runs) == 0.5
    assert m.unique_at_n(["x", "x", "y"]) == 2


def test_tar_answer_uses_parser():
    runs = [["Answer: 4", "answer: 4"], ["5", "6"]]
    assert m.tar_answer(runs, lambda s: s.split()[-1]) == 0.5


def test_first_divergence():
    assert m.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert m.first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert m.first_divergence([1, 2], [1, 2, 3]) == 2


def test_flip_report():
    r = m.flip_report(old=["a", "b", "c"], new=["a", "x", "c2"], gold=["a", "b", "c2"])
    assert (r.negative_flips, r.positive_flips, r.changed) == (1, 1, 2)
    assert r.negative_flip_rate == pytest.approx(1 / 3)


def test_ece_perfect_and_bad():
    assert m.expected_calibration_error([1.0, 1.0], [True, True]) == 0.0
    assert m.expected_calibration_error([0.9, 0.9], [False, False]) == pytest.approx(0.9)


def test_majority_share():
    assert m.majority_share(["a", "a", "b", "c"]) == 0.5
