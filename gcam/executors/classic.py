"""Classic ML executors (plan item v): a GBDT for tabular decisions, a forecaster, a TF-IDF
retriever. Zero LLM tokens; deterministic on a fixed CPU stack (D1); calibrated abstention so a
cascade can hand the uncertain cases to a model that reads the text.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from ..domain import EXPENSE_CATS, POLICY, RISKY_VENDORS
from .base import ExecResult, NotSupported


def conformal_threshold(conf: list[float], correct: list[bool], target: float) -> float:
    """Smallest confidence threshold whose answered accuracy on a held-out split meets ``target``.

    Returns +inf (always abstain) when no threshold reaches the target.
    """
    pairs = sorted(zip(conf, correct))
    for i, (c, _) in enumerate(pairs):
        kept = [ok for cc, ok in pairs[i:] if cc >= c]
        if kept and sum(kept) / len(kept) >= target:
            return float(c)
    return float("inf")


def expense_features(rows: list[dict]) -> np.ndarray:
    feats = []
    for r in rows:
        f = [r["amount"], np.log1p(r["amount"]), float(r["amount"] > 500), float(r["has_receipt"]), r["tenure_months"],
             float(r["weekend"]), float(r["round_amount"]), float(r["duplicate_7d"]), float(r["vendor"] in RISKY_VENDORS)]
        f += [float(r["category"] == c) for c in EXPENSE_CATS]
        feats.append(f)
    return np.asarray(feats, dtype=np.float64)


class ExpenseGBDT:
    """Gradient-boosted trees on the structured expense fields (it never reads the memo)."""

    name = "ml.gbdt"

    def __init__(self, seed: int = 0):
        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed)
        self.tau = float("inf")
        self.version = "ml.gbdt/1"

    def fit(self, rows: list[dict]) -> "ExpenseGBDT":
        self.model.fit(expense_features(rows), np.asarray([r["needs_review"] for r in rows]))
        return self

    def proba(self, rows: list[dict]) -> np.ndarray:
        return self.model.predict_proba(expense_features(rows))[:, 1]

    def calibrate(self, rows: list[dict], target: float) -> float:
        p = self.proba(rows)
        conf = np.maximum(p, 1 - p).tolist()
        correct = [(pi > 0.5) == r["needs_review"] for pi, r in zip(p, rows)]
        self.tau = conformal_threshold(conf, correct, target)
        return self.tau

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        if op != "decide.expense_review":
            raise NotSupported(op)
        p = float(self.proba([payload["row"]])[0])
        conf = max(p, 1 - p)
        value = {"expense_id": payload["row"]["expense_id"], "needs_review": p > 0.5}
        return ExecResult(forced or conf >= self.tau, value, conf, detail={"p_review": round(p, 4)})


class OrderForecaster:
    """Ridge on weekday + trend, chosen against seasonal-naive by a 28-day backtest; intervals from
    backtest residual quantiles."""

    name = "ml.forecast"
    version = "ml.forecast/1"

    def __init__(self, daily: pd.Series):
        self.daily = daily

    @staticmethod
    def _design(dates: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
        dow = np.eye(7)[dates.weekday]
        trend = ((dates - t0).days.values / 30.0)[:, None]
        return np.hstack([dow, trend])

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        if op != "forecast.orders":
            raise NotSupported(op)
        now = pd.Timestamp(datetime.fromisoformat(payload["now"]).date())
        y = self.daily[self.daily.index < now]
        t0 = y.index[0]
        train, test = y.iloc[:-28], y.iloc[-28:]
        ridge = Ridge(alpha=1.0).fit(self._design(train.index, t0), train.values)
        pred_ridge = ridge.predict(self._design(test.index, t0))
        pred_naive = y.shift(7).iloc[-28:].values
        mae_ridge = float(np.mean(np.abs(pred_ridge - test.values)))
        mae_naive = float(np.mean(np.abs(pred_naive - test.values)))
        use_ridge = mae_ridge <= mae_naive
        resid = (pred_ridge if use_ridge else pred_naive) - test.values
        lo_q, hi_q = np.quantile(resid, [0.1, 0.9])
        future = pd.date_range(now, periods=payload["days"], freq="D")
        if use_ridge:
            model = Ridge(alpha=1.0).fit(self._design(y.index, t0), y.values)
            point = model.predict(self._design(future, t0))
        else:
            point = np.array([y[y.index.weekday == d.weekday()].iloc[-1] for d in future], dtype=float)
        rows = [{"date": d.date().isoformat(), "point": round(float(p), 1),
                 "lo": round(float(p - hi_q), 1), "hi": round(float(p - lo_q), 1)} for d, p in zip(future, point)]
        value = {"model": "ridge(weekday+trend)" if use_ridge else "seasonal_naive",
                 "backtest_mae": round(mae_ridge if use_ridge else mae_naive, 2),
                 "baseline_mae": round(mae_naive, 2), "interval": "80%", "forecast": rows}
        return ExecResult(True, value, 1.0)


class PolicyTfidf:
    """Sparse lexical retrieval over the policy sections; abstains when the top two are close."""

    name = "ml.tfidf"
    version = "ml.tfidf/1"

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.docs = [f"{s.title}. {s.text}" for s in POLICY]
        self.mat = self.vec.fit_transform(self.docs)
        self.tau = 0.0

    def scores(self, query: str) -> np.ndarray:
        return (self.mat @ self.vec.transform([query]).T).toarray().ravel()

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        if op != "retrieve.policy":
            raise NotSupported(op)
        s = self.scores(payload["query"])
        order = np.argsort(-s, kind="stable")
        margin = float(s[order[0]] - s[order[1]])
        sec = POLICY[int(order[0])]
        value = {"id": sec.id, "title": sec.title, "score": round(float(s[order[0]]), 4)}
        return ExecResult(forced or (s[order[0]] > 0 and margin >= self.tau), value, margin)
