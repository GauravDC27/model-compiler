import math

import pytest

torch = pytest.importorskip("torch")
from dmc import moe_probe as mp  # noqa: E402


def test_stable_topk_breaks_ties_by_lowest_index():
    logits = torch.tensor([[1.0, 3.0, 3.0, 3.0, 0.0]])
    assert mp.stable_topk(logits, 2).tolist() == [[1, 2]]


def test_margins_and_flips():
    a = torch.tensor([[[5.0, 4.0, 3.9, 0.0]]])
    b = torch.tensor([[[5.0, 3.8, 3.9, 0.0]]])
    assert mp.margins(a, 2).item() == pytest.approx(0.1)
    assert mp.flips(a, b, 2).item() is True
    assert mp.flips(a, a, 2).item() is False


def test_roc_auc():
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([False, False, True, True])
    assert mp.roc_auc(scores, labels) == 1.0
    assert mp.roc_auc(-scores, labels) == 0.0
    assert math.isnan(mp.roc_auc(scores, torch.zeros(4, dtype=torch.bool)))


def test_task_specificity_extremes():
    # 1 layer, 4 experts: tasks use disjoint experts -> NMI 1; identical usage -> NMI 0.
    disjoint = {"a": torch.tensor([[5.0, 5.0, 0, 0]]), "b": torch.tensor([[0, 0, 5.0, 5.0]])}
    same = {"a": torch.tensor([[1.0, 2, 3, 4]]), "b": torch.tensor([[1.0, 2, 3, 4]])}
    assert mp.task_specificity(disjoint).item() == pytest.approx(1.0)
    assert mp.task_specificity(same).item() == pytest.approx(0.0, abs=1e-9)


def test_expert_usage_respects_mask():
    logits = torch.tensor([[[[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]]])  # L=1,B=1,T=2,E=3
    mask = torch.tensor([[1.0, 0.0]])
    assert mp.expert_usage(logits, 1, mask).tolist() == [[1.0, 0.0, 0.0]]
