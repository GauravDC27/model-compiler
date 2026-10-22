"""System-1 executors (plan item iv): answer from one encoder pass or one hub prefill, with no
generated tokens, and abstain upward when calibrated confidence is low.

* ``EncoderHeads`` — all-MiniLM-L6-v2 sentence embeddings (mean pooled, normalized) with a
  logistic-regression head per task: claim reason, tool, policy section, and the expense memo's
  signal (approve / review / neither), which decides a borderline report the GBDT hands off;
  dense retrieval over the policy sections when no section head is trained.
* ``HubProbe`` — a logistic-regression decision head on the hub's own hidden states (a probe at
  a middle layer): the decision that would otherwise cost a full prompt costs one prefill of the
  bare text.

Each input is encoded alone at its own length (no padding, no batch-mates), the CPU analogue of
a batch-invariant lane.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from ..domain import POLICY
from .base import ExecResult, NotSupported
from .classic import conformal_threshold

ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder:
    def __init__(self, model_id: str = ENCODER_ID):
        from transformers import AutoModel, AutoTokenizer

        torch.use_deterministic_algorithms(True)
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
        self._cache: dict[str, tuple[np.ndarray, int]] = {}

    def embed(self, text: str) -> tuple[np.ndarray, int]:
        if text not in self._cache:
            enc = self.tok(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state[0]
            v = h.mean(0)
            v = (v / v.norm()).numpy().astype(np.float64)
            self._cache[text] = (v, int(enc.input_ids.shape[1]))
        return self._cache[text]


class LinearHead:
    """Multinomial logistic regression with a conformal abstention threshold."""

    def __init__(self, labels: tuple[str, ...], C: float = 4.0):
        self.labels = labels
        self.clf = LogisticRegression(C=C, max_iter=2000)
        self.tau = float("inf")

    def fit(self, X: np.ndarray, y: list[str]) -> "LinearHead":
        self.clf.fit(X, [self.labels.index(v) for v in y])
        return self

    def predict(self, x: np.ndarray) -> tuple[str, float]:
        p = self.clf.predict_proba(x[None])[0]
        k = int(np.argmax(p))
        return self.labels[int(self.clf.classes_[k])], float(p[k])

    def calibrate(self, X: np.ndarray, y: list[str], target: float) -> float:
        preds = [self.predict(x) for x in X]
        self.tau = conformal_threshold([c for _, c in preds], [lab == t for (lab, _), t in zip(preds, y)], target)
        return self.tau


class EncoderHeads:
    name = "s1.encoder"
    version = "s1.encoder/2"

    def __init__(self, encoder: Encoder, heads: dict[str, LinearHead], retrieval_tau: float = 0.0, gbdt=None):
        self.encoder = encoder
        self.heads = heads  # op (or "memo.signal") -> head
        self.retrieval_tau = retrieval_tau
        self.gbdt = gbdt  # the structured-field model a neutral memo defers to
        self.section_vecs = np.stack([encoder.embed(f"{s.title}. {s.text}")[0] for s in POLICY])

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        if op == "retrieve.policy" and op in self.heads:
            v, n = self.encoder.embed(payload["query"])
            head = self.heads[op]
            label, conf = head.predict(v)
            sec = next(s for s in POLICY if s.id == label)
            value = {"id": sec.id, "title": sec.title, "score": round(conf, 4)}
            return ExecResult(forced or conf >= head.tau, value, conf, usage={"encoder": n})
        if op == "retrieve.policy":
            v, n = self.encoder.embed(payload["query"])
            s = self.section_vecs @ v
            order = np.argsort(-s, kind="stable")
            margin = float(s[order[0]] - s[order[1]])
            sec = POLICY[int(order[0])]
            value = {"id": sec.id, "title": sec.title, "score": round(float(s[order[0]]), 4)}
            return ExecResult(forced or margin >= self.retrieval_tau, value, margin, usage={"encoder": n})
        if op not in self.heads and not (op == "decide.expense_review" and "memo.signal" in self.heads):
            raise NotSupported(op)
        if op == "decide.expense_review":  # P5: on a borderline report the memo decides
            row = payload["row"]
            v, n = self.encoder.embed(row["memo"])
            head = self.heads["memo.signal"]
            signal, conf = head.predict(v)
            review = signal == "review" if signal != "neither" else bool(self.gbdt.proba([row])[0] > 0.5)
            value = {"expense_id": row["expense_id"], "needs_review": review}
            return ExecResult(forced or conf >= head.tau, value, conf, usage={"encoder": n}, detail={"signal": signal})
        v, n = self.encoder.embed(payload["text"])
        head = self.heads[op]
        label, conf = head.predict(v)
        return ExecResult(forced or conf >= head.tau, label, conf, usage={"encoder": n})


class HubProbe:
    """Decision head on hub hidden states (layer ``layer``, mean over the bare text's tokens)."""

    name = "s1.hubprobe"
    version = "s1.hubprobe/1"

    def __init__(self, hub, layer: int, heads: dict[str, LinearHead]):
        self.hub = hub
        self.layer = layer
        self.heads = heads
        self._cache: dict[str, tuple[np.ndarray, int]] = {}

    def features(self, text: str) -> tuple[np.ndarray, int]:
        if text not in self._cache:
            self._cache[text] = self.hub.hidden_mean(text, self.layer)
        return self._cache[text]

    def run(self, op: str, payload: dict, *, forced: bool) -> ExecResult:
        if op not in self.heads:
            raise NotSupported(op)
        v, n = self.features(payload["text"])
        label, conf = self.heads[op].predict(v)
        return ExecResult(forced or conf >= self.heads[op].tau, label, conf, usage={"hub": n})
