"""System-1 / code / System-2 cascade compiled by the scheduler.
Pre-registration: 07_system1_cascade_prereg.md."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dmc.oracle import generate  # noqa: E402
from dmc.profiles.decide import SYSTEM as DECIDE_SYSTEM  # noqa: E402
from dmc.profiles.decide import DecideProfile, label_orderings, render_user  # noqa: E402
from dmc.runtime import HFBackend  # noqa: E402
from dmc.scheduler import CompileError, Executor, Node, compile_schedule  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
OP = "decide.classify"
LAYERS = (6, 12, 18, 23)
SMOKE = bool(os.environ.get("DMC_SMOKE"))
OUT = ROOT / f"experiments/results/07_system1_cascade{'_smoke' if SMOKE else ''}.json"

# Template-0 grammars of the oracle families: the "compiler" knows only the canonical forms.
GRAMMAR = [
    (re.compile(r"^Is (\d+) greater than (\d+)\?"), lambda m: int(m[1]) > int(m[2])),
    (re.compile(r"^Is (\d+) an even number\?"), lambda m: int(m[1]) % 2 == 0),
    (re.compile(r"^Our policy allows refunds within (\d+) days of purchase\. The customer bought the item (\d+) days ago\."),
     lambda m: int(m[2]) < int(m[1])),
    (re.compile(r"^Does the word '(\w+)' contain the letter '(\w)'\?"), lambda m: m[2] in m[1]),
    (re.compile(r"^Is (\d+) \+ (\d+) equal to (\d+)\?"), lambda m: int(m[1]) + int(m[2]) == int(m[3])),
    (re.compile(r"^Expenses under \$(\d+) are auto-approved\. This expense is \$(\d+)\."),
     lambda m: int(m[2]) < int(m[1])),
]


def run_code(q: str) -> str | None:
    for rx, fn in GRAMMAR:
        m = rx.match(q)
        if m:
            return "yes" if fn(m) else "no"
    return None


def features(b: HFBackend, q: str) -> tuple[torch.Tensor, int]:
    ids = b._ids(q)
    with torch.no_grad():
        hs = b.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    return torch.stack([hs[layer + 1][0, -1].float() for layer in LAYERS]), len(ids)


def fit_logreg(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = (x - mu) / sd
    w = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    b0 = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b0], max_iter=200, line_search_fn="strong_wolfe")
    zz, yy = z.double(), y.double()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(zz @ w + b0, yy) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (w.detach(), b0.detach()), (mu, sd)


def predict(model, x: torch.Tensor) -> torch.Tensor:
    (w, b0), (mu, sd) = model
    return torch.sigmoid(((x - mu) / sd).double() @ w + b0)


def wilson(p: float, n: int) -> tuple[float, float]:
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    t0 = time.time()
    b = HFBackend(MODEL, dtype="float32")
    train = generate(2 if SMOKE else 60, seed=100)
    calib = generate(2 if SMOKE else 10, seed=7)
    test = generate(1 if SMOKE else 20, seed=0)

    def feats(items):
        fs, ns = zip(*(features(b, it.question) for it in items))
        return torch.stack(fs), list(ns)

    ftr, _ = feats(train)
    fca, nca = feats(calib)
    fte, nte = feats(test)
    ytr = torch.tensor([it.answer == "yes" for it in train], dtype=torch.float64)
    yca = [it.answer == "yes" for it in calib]
    print(f"features done {time.time() - t0:.0f}s", flush=True)

    # Layer choice on calibration (forced accuracy).
    best = None
    for li, layer in enumerate(LAYERS):
        m = fit_logreg(ftr[:, li], ytr)
        p = predict(m, fca[:, li])
        acc = sum((pi > 0.5) == yi for pi, yi in zip(p.tolist(), yca)) / len(yca)
        if best is None or acc > best[0]:
            best = (acc, li, layer, m)
    s1_acc_all, li, layer, probe = best
    p_ca = predict(probe, fca[:, li]).tolist()
    conf = [max(p, 1 - p) for p in p_ca]
    tau = 1.0
    for t in sorted(set(conf)):
        ans = [(pc > 0.5) == yc for pc, yc, c in zip(p_ca, yca, conf) if c >= t]
        if ans and sum(ans) / len(ans) >= 0.95:
            tau = t
            break
    covered = [((pc > 0.5) == yc) for pc, yc, c in zip(p_ca, yca, conf) if c >= tau]
    s1_cov = len(covered) / len(yca)
    s1_acc = sum(covered) / len(covered) if covered else 0.0

    code_ca = [run_code(it.question) for it in calib]
    code_cov = sum(a is not None for a in code_ca) / len(calib)
    code_acc = (sum(a == it.answer for a, it in zip(code_ca, calib) if a is not None)
                / max(1, sum(a is not None for a in code_ca)))

    prof = DecideProfile(b, abstain_margin=0.0)

    def s2_tokens(it) -> int:
        op = it.op()
        return sum(len(b._ids(b.render_chat(DECIDE_SYSTEM, render_user(op, o)))) for o in label_orderings(op.labels))

    s2_ca = [prof.run(it.op()).label == it.answer for it in calib]
    s2_acc_all = sum(s2_ca) / len(s2_ca)
    s2_tok = sum(s2_tokens(it) for it in calib) / len(calib)
    s1_tok = sum(nca) / len(nca)
    executors = [
        Executor("code.canonical", "tool", frozenset({OP}), 0, 0, 1, {OP: code_acc}, coverage=code_cov),
        Executor("s1.probe", "system1", frozenset({OP}), round(s1_tok), 0, 2, {OP: s1_acc},
                 coverage=s1_cov, accuracy_all={OP: s1_acc_all}),
        Executor("s2.decide", "system2", frozenset({OP}), round(s2_tok), 0, 10, {OP: s2_acc_all}),
    ]
    result = {"model": MODEL, "probe_layer": layer, "tau": tau, "calibration": {
        "code": {"coverage": code_cov, "accuracy": code_acc},
        "s1_probe": {"coverage": s1_cov, "accuracy_answered": s1_acc, "accuracy_all": s1_acc_all,
                     "mean_tokens": s1_tok},
        "s2_decide": {"accuracy_all": s2_acc_all, "mean_tokens": s2_tok}}}
    print(result["calibration"], flush=True)

    # Realized per-item outcomes on test for each executor.
    p_te = predict(probe, fte[:, li]).tolist()
    s2_te = [prof.run(it.op()).label for it in test]
    s2_te_tok = [s2_tokens(it) for it in test]
    out = {}
    for req in (0.6, 0.7, 0.8):
        try:
            sch = compile_schedule([Node(f"q{i}", OP, required_accuracy=req) for i in range(len(test))],
                                   executors, token_budget=10**9)
        except CompileError as e:
            out[str(req)] = {"compile_error": str(e)}
            continue
        ladder = sch.assignments[0].ladder
        worst_of = {a.node: a.worst_case_tokens for a in sch.assignments}
        correct, tokens, worst_ok = [], [], True
        for i, it in enumerate(test):
            spent, answer = 0, None
            for name in ladder:
                if name == "code.canonical":
                    answer = run_code(it.question)
                elif name == "s1.probe":
                    spent += nte[i]
                    c = max(p_te[i], 1 - p_te[i])
                    if c >= tau or name == ladder[-1]:
                        answer = "yes" if p_te[i] > 0.5 else "no"
                else:
                    spent += s2_te_tok[i]
                    answer = s2_te[i]
                if answer is not None:
                    break
            correct.append(answer == it.answer)
            tokens.append(spent)
            worst_ok &= spent <= worst_of[f"q{i}"]
        acc = sum(correct) / len(correct)
        pred_acc = sch.assignments[0].expected_accuracy
        lo, hi = wilson(pred_acc, len(test))
        out[str(req)] = {
            "ladder": ladder, "predicted_accuracy": pred_acc, "realized_accuracy": acc,
            "realized_within_predicted_wilson95": lo <= acc <= hi,
            "predicted_tokens_per_decision": sch.expected_tokens / len(test),
            "realized_tokens_per_decision": sum(tokens) / len(tokens),
            "token_prediction_error": (sum(tokens) / len(tokens)) / max(1e-9, sch.expected_tokens / len(test)) - 1,
            "worst_case_respected": worst_ok, "schedule_hash": sch.hash,
        }
    base_acc = sum(a == it.answer for a, it in zip(s2_te, test)) / len(test)
    result["always_s2"] = {"accuracy": base_acc, "tokens_per_decision": sum(s2_te_tok) / len(test)}
    result["compiled"] = out
    result["elapsed_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
