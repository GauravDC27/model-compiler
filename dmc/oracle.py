"""Synthetic-oracle generator: decision prompts with known answers ("Csmith for prompts", R20).

Each item belongs to a paraphrase group: the same underlying question (same parameters,
same correct answer) rendered with different surface templates. That gives, from one
generator, accuracy, paraphrase invariance and consistent-correct rates with enough items
for confidence intervals. Generation is a pure function of the seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .ir import Context, DecideClassify

LABELS = ("yes", "no")
SUFFIX = " Answer yes or no."
WORDS = ("banana", "orchard", "window", "planet", "silver", "garden", "rocket", "pencil",
         "harbor", "meadow", "candle", "forest", "bridge", "button", "marble", "jacket")


@dataclass(frozen=True)
class OracleItem:
    family: str
    group: int
    paraphrase: int
    question: str
    answer: str  # "yes" | "no"

    def op(self) -> DecideClassify:
        ctx = Context(now=None, locale="en-US", timezone=None, session_state_hash=None)
        return DecideClassify(context=ctx, ambiguities=(), question=self.question, labels=LABELS,
                              policy=None, evidence=())


def _compare(r: random.Random, want_yes: bool):
    a, b = r.sample(range(10, 100), 2)
    if (a > b) != want_yes:
        a, b = b, a
    return [f"Is {a} greater than {b}?", f"Is the number {a} larger than the number {b}?",
            f"Compare {a} and {b}: is {a} the bigger one?"], a > b


def _even(r: random.Random, want_yes: bool):
    n = r.randrange(10, 500)
    if (n % 2 == 0) != want_yes:
        n += 1
    return [f"Is {n} an even number?", f"Is the number {n} even?",
            f"Can {n} be divided by 2 with no remainder?"], n % 2 == 0


def _refund(r: random.Random, want_yes: bool):
    window = r.choice((7, 14, 30, 60, 90))
    days = r.randrange(1, window) if want_yes else r.randrange(window + 1, window * 2 + 2)
    return [
        f"Our policy allows refunds within {window} days of purchase. The customer bought the item {days} days ago. Should we issue a refund?",
        f"Refunds are allowed for {window} days after purchase. This purchase was {days} days ago. Is a refund allowed?",
        f"The item was bought {days} days ago and the refund window is {window} days. Is the customer eligible for a refund?",
    ], days < window


def _letter(r: random.Random, want_yes: bool):
    word = r.choice(WORDS)
    present = sorted(set(word))
    absent = sorted(set("abcdefghijklmnopqrstuvwxyz") - set(word))
    c = r.choice(present if want_yes else absent)
    return [f"Does the word '{word}' contain the letter '{c}'?",
            f"Is the letter '{c}' in the word '{word}'?",
            f"Looking at the word '{word}', does the letter '{c}' appear in it?"], c in word


def _sum(r: random.Random, want_yes: bool):
    a, b = r.randrange(2, 50), r.randrange(2, 50)
    s = a + b if want_yes else a + b + r.choice((-3, -2, -1, 1, 2, 3))
    return [f"Is {a} + {b} equal to {s}?", f"Does {a} plus {b} make {s}?",
            f"If you add {a} and {b}, do you get {s}?"], a + b == s


def _expense(r: random.Random, want_yes: bool):
    limit = r.choice((50, 100, 200, 500, 1000))
    x = r.randrange(1, limit) if want_yes else r.randrange(limit + 1, limit * 3)
    return [
        f"Expenses under ${limit} are auto-approved. This expense is ${x}. Should it be auto-approved?",
        f"The auto-approval limit is ${limit}. An expense of ${x} was submitted. Is it auto-approved?",
        f"Auto-approve any expense below ${limit}. Expense amount: ${x}. Auto-approve it?",
    ], x < limit


FAMILIES = {"compare": _compare, "even": _even, "refund": _refund, "letter": _letter,
            "sum": _sum, "expense": _expense}


def generate(groups_per_family: int = 20, seed: int = 0) -> list[OracleItem]:
    """Balanced yes/no items: ``groups_per_family`` groups x 3 paraphrases per family."""
    r = random.Random(seed)
    items = []
    for fam, make in FAMILIES.items():
        for g in range(groups_per_family):
            texts, truth = make(r, want_yes=(g % 2 == 0))
            answer = "yes" if truth else "no"
            for p, t in enumerate(texts):
                items.append(OracleItem(fam, g, p, t + SUFFIX, answer))
    return items
