"""Acme Ops: the deterministic synthetic company that GCAM v1 works for.

Everything is a pure function of a seed, so every goal, label and table is reproducible and
its snapshot hash can enter receipts (R23). The world has:

* customers and 180 days of orders (weekly seasonality, a trend, per-category refund rates);
* expense reports whose review label depends on structured risk and, near the threshold,
  on what the memo says (so a table model is unsure exactly where reading the memo helps);
* a written refund/expense policy split into sections;
* generators for the free-text inputs agents see: refund claims, support requests that need
  a tool, reply pairs to compare, and cost-estimate word problems.

Text generators keep separate *training* and *held-out* phrasings, so evaluation measures
out-of-template behaviour, not memorized templates.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from dmc.canonical import content_hash

NOW = "2026-10-21T10:00:00+00:00"
NOW_DT = datetime.fromisoformat(NOW)
CATEGORIES = ("electronics", "apparel", "home", "toys", "grocery")
REFUND_RATE = {"electronics": 0.09, "apparel": 0.15, "home": 0.06, "toys": 0.05, "grocery": 0.03}
FIRST = ("Maria", "James", "Aisha", "Wei", "Lucas", "Priya", "Tom", "Elena", "Kofi", "Hana", "Diego",
         "Sara", "Omar", "Mei", "Noah", "Ines", "Ravi", "Zoe", "Ivan", "Leila")
LAST = ("Lopez", "Smith", "Khan", "Chen", "Silva", "Patel", "Brown", "Rossi", "Mensah", "Sato", "Garcia",
        "Novak", "Haddad", "Wong", "Miller", "Costa", "Iyer", "Weber", "Petrov", "Nasser")
PRODUCTS = {
    "electronics": ("headphones", "blender", "phone charger", "smart speaker", "laptop stand", "monitor"),
    "apparel": ("jacket", "pair of jeans", "running shoes", "wool sweater", "rain coat", "dress"),
    "home": ("desk lamp", "throw blanket", "ceramic vase", "kitchen knife set", "pillow", "curtain set"),
    "toys": ("puzzle", "toy robot", "board game", "plush bear", "kite", "building set"),
    "grocery": ("coffee beans", "olive oil", "tea sampler", "spice kit", "honey jar", "pasta box"),
}
CLAIM_REASONS = ("damaged", "not_received", "wrong_item", "changed_mind")
TOOLS = ("get_order_status", "schedule_callback", "cancel_order", "update_address", "search_policy",
         "open_refund_case")
TOOL_CAPABILITY = {
    "get_order_status": "read", "search_policy": "read", "schedule_callback": "write",
    "update_address": "write", "open_refund_case": "write", "cancel_order": "destructive",
}


@dataclass(frozen=True)
class PolicySection:
    id: str
    title: str
    text: str


POLICY = (
    PolicySection("P1", "Refund window", "Delivered items may be returned for a full refund within 30 days of delivery "
                  "if the customer changed their mind. Grocery items cannot be returned for a change of mind."),
    PolicySection("P2", "Damaged or defective items", "Items that arrive damaged, broken or defective are refunded "
                  "within 90 days of delivery, including grocery items. A photo of the damage is requested."),
    PolicySection("P3", "Items not received", "If an order has not been delivered 10 days after it was placed, "
                  "the customer receives a full refund or a replacement. Before 10 days, ask them to wait."),
    PolicySection("P4", "Wrong item shipped", "If we shipped the wrong item, colour or size, the customer gets a refund "
                  "or exchange within 60 days of delivery, and we pay return shipping."),
    PolicySection("P5", "Expense approval", "Expense reports under 500 dollars with a receipt are approved automatically. "
                  "Reports without a receipt, over 500 dollars, or for personal items need manual review."),
    PolicySection("P6", "Callbacks", "Customers can ask for a phone callback. Callbacks are scheduled between 9am and 6pm "
                  "on weekdays, in the customer's local time."),
    PolicySection("P7", "Address changes", "Shipping addresses can be changed until the order ships. After shipping, "
                  "the customer must contact the carrier."),
    PolicySection("P8", "Cancellations", "Orders can be cancelled before they ship. Cancelling is permanent and refunds "
                  "the full amount to the original payment method."),
)


@dataclass(frozen=True)
class World:
    customers: pd.DataFrame
    orders: pd.DataFrame
    expenses: pd.DataFrame
    daily_orders: pd.Series
    policy: tuple[PolicySection, ...] = POLICY
    seed: int = 0
    _hash: list = field(default_factory=list, repr=False, compare=False)

    def snapshot_hash(self) -> str:
        """Content hash of every table: snapshotted inputs enter receipts (R23)."""
        if not self._hash:
            self._hash.append(content_hash({
                "customers": self.customers.to_csv(index=False),
                "orders": self.orders.to_csv(index=False),
                "expenses": self.expenses.to_csv(index=False),
                "daily_orders": self.daily_orders.to_csv(),
                "policy": [s.__dict__ for s in self.policy],
            }))
        return self._hash[0]

    def order(self, order_id: str) -> dict | None:
        rows = self.orders[self.orders.order_id == order_id]
        return None if rows.empty else rows.iloc[0].to_dict()

    def customer(self, customer_id: str) -> dict | None:
        rows = self.customers[self.customers.customer_id == customer_id]
        return None if rows.empty else rows.iloc[0].to_dict()


def build_world(seed: int = 7) -> World:
    rng = np.random.default_rng(seed)
    n_cust = 240
    customers = pd.DataFrame({
        "customer_id": [f"C-{1000 + i}" for i in range(n_cust)],
        "name": [f"{FIRST[i % len(FIRST)]} {LAST[(i * 7 + i // len(FIRST)) % len(LAST)]}" for i in range(n_cust)],
        "tier": rng.choice(["standard", "plus", "business"], size=n_cust, p=[0.7, 0.2, 0.1]),
        "tenure_months": rng.integers(1, 96, size=n_cust),
    })

    days = 180
    start = (NOW_DT - timedelta(days=days)).date()
    dates = [start + timedelta(days=d) for d in range(days)]
    weekday_effect = np.array([0, 0, 0, 1, 3, 6, 4])  # Mon..Sun: weekend peak
    lam = 14 + 0.03 * np.arange(days) + weekday_effect[[d.weekday() for d in dates]]
    counts = rng.poisson(lam)
    daily = pd.Series(counts, index=pd.to_datetime(dates), name="orders")

    rows = []
    oid = 5000
    for d, c in zip(dates, counts):
        for _ in range(int(c)):
            cat = CATEGORIES[int(rng.choice(5, p=[0.25, 0.25, 0.2, 0.15, 0.15]))]
            amount = float(np.round(np.exp(rng.normal(3.6 if cat != "grocery" else 2.8, 0.6)), 2))
            age = (NOW_DT.date() - d).days
            ship_days = int(rng.integers(2, 7))
            if age > ship_days + 1:
                status = "delivered" if rng.random() > 0.015 else "lost_in_transit"
            else:
                status = "shipped" if age >= 2 else "processing"
            delivered = (d + timedelta(days=ship_days)).isoformat() if status == "delivered" else None
            refunded = bool(status == "delivered" and rng.random() < REFUND_RATE[cat])
            rows.append({
                "order_id": f"A-{oid}", "customer_id": customers.customer_id[int(rng.integers(n_cust))],
                "category": cat, "product": PRODUCTS[cat][int(rng.integers(len(PRODUCTS[cat])))],
                "amount": amount, "order_date": d.isoformat(), "delivered_date": delivered,
                "status": status, "refunded": refunded,
            })
            oid += 1
    orders = pd.DataFrame(rows)
    expenses = _expenses(rng, customers)
    return World(customers=customers, orders=orders, expenses=expenses, daily_orders=daily, seed=seed)


EXPENSE_CATS = ("meals", "travel", "software", "office", "gifts")
VENDORS = {"meals": ("Bistro 21", "Cafe Lumen", "Noodle Bar"), "travel": ("SkyJet", "MetroRail", "CityCab"),
           "software": ("CloudDesk", "CodeForge", "DataPeak"), "office": ("PaperCo", "DeskWorks", "InkHub"),
           "gifts": ("GiftHub", "FlowerPost", "CardShop")}
RISKY_VENDORS = {"GiftHub", "CardShop", "CityCab"}
MEMO_CLEAR = {"meals": "Client lunch with {who}, receipt attached", "travel": "Train to client site for {who} workshop",
              "software": "Annual licence renewal for the team", "office": "Printer paper and toner for the office",
              "gifts": "Thank-you flowers for a client after the launch"}
MEMO_RISK_UP = ("receipt lost, amount from memory", "gift cards for the team holiday", "personal laptop sleeve, will reimburse later",
                "split across two reports to stay under the limit", "cash payment, no invoice available")
MEMO_RISK_DOWN = ("pre-approved by finance, approval ref FIN-{n}", "client-billable, recharged on invoice INV-{n}",
                  "budgeted team offsite, approved in the quarterly plan", "conference fee, registration confirmation attached",
                  "replacement for a broken office monitor, IT ticket IT-{n}")


def _expenses(rng: np.random.Generator, customers: pd.DataFrame) -> pd.DataFrame:
    n = 1600
    rows = []
    for i in range(n):
        cat = EXPENSE_CATS[int(rng.integers(5))]
        vendor = VENDORS[cat][int(rng.integers(3))]
        amount = float(np.round(np.exp(rng.normal(4.6 if cat in ("travel", "software") else 4.0, 0.8)), 2))
        has_receipt = bool(rng.random() > 0.15)
        tenure = int(rng.integers(1, 120))
        weekend = bool(rng.random() < 0.18)
        round_amount = bool(rng.random() < 0.12)
        if round_amount:
            amount = float(round(amount, -1) or 10.0)
        duplicate = bool(rng.random() < 0.05)
        risk = (1.6 * (amount > 500) + 1.4 * (not has_receipt) + 1.0 * (vendor in RISKY_VENDORS) + 0.6 * weekend
                + 0.7 * round_amount + 1.5 * duplicate + 0.5 * (tenure < 6) + 0.4 * (cat == "gifts")
                + rng.normal(0, 0.35))
        who = FIRST[int(rng.integers(len(FIRST)))]
        memo = MEMO_CLEAR[cat].format(who=who)
        memo_shift = 0.0
        # Near the threshold the memo carries the deciding information (a reviewer reads it).
        if 1.2 < risk < 2.6:
            if rng.random() < 0.5:
                memo, memo_shift = MEMO_RISK_UP[int(rng.integers(5))], 1.5
            else:
                memo, memo_shift = MEMO_RISK_DOWN[int(rng.integers(5))].format(n=int(rng.integers(1000, 9999))), -1.5
        needs_review = bool(risk + memo_shift > 1.9)
        d = (NOW_DT - timedelta(days=int(rng.integers(0, 180)))).date().isoformat()
        rows.append({"expense_id": f"E-{20000 + i}", "date": d, "submitter": customers.name[int(rng.integers(len(customers)))],
                     "category": cat, "vendor": vendor, "amount": amount, "has_receipt": has_receipt,
                     "tenure_months": tenure, "weekend": weekend, "round_amount": round_amount,
                     "duplicate_7d": duplicate, "memo": memo, "needs_review": needs_review})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- refund claims

CLAIM_TEMPLATES = {
    "damaged": {
        "train": ("The {p} arrived damaged.", "My {p} came broken.", "The {p} is defective and does not work.",
                  "When I opened the box the {p} was smashed.", "The {p} stopped working on the first day.",
                  "There is a big crack in the {p}.", "The {p} was torn when it arrived."),
        "test": ("Out of the box the {p} has a dent along one side and makes a grinding noise.",
                 "Glass shards everywhere, the {p} must have been dropped in transit.",
                 "It powers on for a second and then dies, this {p} is faulty."),
    },
    "not_received": {
        "train": ("My {p} never arrived.", "I have not received my {p}.", "The {p} is still not here.",
                  "Where is my {p}? It has not been delivered.", "The package with the {p} never showed up.",
                  "I am still waiting for the {p} to arrive."),
        "test": ("Tracking claims it was dropped off but there was nothing on my porch, no {p}.",
                 "Two weeks and counting, the {p} is nowhere to be seen.",
                 "The courier says delivered, my mailbox says otherwise, still no {p}."),
    },
    "wrong_item": {
        "train": ("You sent me the wrong {p}.", "I ordered a different {p}, this is the wrong one.",
                  "The {p} is the wrong size.", "I got a {p} in the wrong colour.", "This is not the {p} I ordered.",
                  "The box contained someone else's {p}."),
        "test": ("I picked medium but the {p} in the parcel is an extra large.",
                 "Ordered the black {p}, received a bright orange one.",
                 "The label says {p} but inside was a completely different product."),
    },
    "changed_mind": {
        "train": ("I changed my mind about the {p}.", "I no longer need the {p}.", "I do not want the {p} anymore.",
                  "I found the {p} cheaper somewhere else.", "I bought the {p} by mistake.",
                  "The {p} is fine but I would like to return it."),
        "test": ("Honestly the {p} works perfectly, it just is not for me after all.",
                 "My sister already gave me a {p}, so this one is surplus.",
                 "Saw a better deal on the {p} at another shop, sending it back."),
    },
}
OPENERS = ("", "Hello, ", "Hi there. ", "Good morning, ", "Excuse me, ")


def _join(opener: str, body: str) -> str:
    """Opener + body, lower-casing the body's first word after a comma (but never "I")."""
    if opener.endswith(", ") and not (body.startswith("I ") or body.startswith("I'")):
        body = body[0].lower() + body[1:]
    return opener + body
CLOSERS = ("", " Please help.", " What can you do?", " Thanks.", " I would like a refund.")


@dataclass(frozen=True)
class Claim:
    text: str
    reason: str
    product: str
    split: str


def claims(n: int, *, split: str, seed: int) -> list[Claim]:
    """Refund claims with their true reason; ``split`` picks training or held-out phrasings."""
    rng = random.Random(f"claims:{split}:{seed}")
    out = []
    for i in range(n):
        reason = CLAIM_REASONS[i % len(CLAIM_REASONS)]
        cat = CATEGORIES[rng.randrange(len(CATEGORIES))]
        product = rng.choice(PRODUCTS[cat])
        body = rng.choice(CLAIM_TEMPLATES[reason][split]).format(p=product)
        text = _join(rng.choice(OPENERS), body) + rng.choice(CLOSERS)
        out.append(Claim(text=text.strip(), reason=reason, product=product, split=split))
    return out


# --------------------------------------------------------------------------- tool requests

TOOL_TEMPLATES = {
    "get_order_status": {
        "train": ("Where is my order {o}?", "What is the status of order {o}?", "Has order {o} shipped yet?",
                  "Can you track order {o} for me?", "When will order {o} arrive?"),
        "test": ("Any news on {o}? I am wondering how far along it is.", "Is {o} on a truck yet or still in the warehouse?"),
    },
    "schedule_callback": {
        "train": ("Please call me {d} at {t}.", "Can someone phone me {d} at {t}?", "I would like a callback {d} at {t}.",
                  "Schedule a call with me {d} at {t}.", "Ring me {d} at {t} please."),
        "test": ("I would rather talk to a human, {d} at {t} works for me.", "Could an agent get in touch by phone {d} around {t}?"),
    },
    "cancel_order": {
        "train": ("Cancel order {o}.", "Please cancel my order {o}.", "I want to cancel order {o}.",
                  "Stop order {o}, I do not want it.", "Cancel {o} before it ships."),
        "test": ("Scrap {o}, I no longer want it sent.", "Kill order {o} please, I ordered it twice."),
    },
    "update_address": {
        "train": ("Change my shipping address to {a}.", "Update my address to {a}.", "Please ship to {a} instead.",
                  "My new address is {a}.", "Send my orders to {a} from now on."),
        "test": ("I moved last week, everything should go to {a} now.", "Deliveries need to go to {a}, not the old place."),
    },
    "search_policy": {
        "train": ("What is your return policy?", "How long do I have to return an item?", "What is the policy on damaged items?",
                  "Do you refund grocery items?", "What are the rules for cancellations?"),
        "test": ("If something shows up broken, what are my options?", "Can I still send back a sweater I bought last month?"),
    },
    "open_refund_case": {
        "train": ("I want a refund for order {o}.", "Please refund order {o}.", "Open a refund for {o}.",
                  "I would like my money back for order {o}.", "Start a refund on order {o}."),
        "test": ("Money back for {o} please, it was not as described.", "I expect to be reimbursed for {o}."),
    },
}
STREETS = ("12 Elm Street", "4 Harbour Road", "77 Queen Avenue", "9 Mill Lane", "301 Pine Court")
DAY_PHRASES = ("tomorrow", "on Friday", "next Monday", "today")
TIME_PHRASES = ("3pm", "10am", "11:30am", "4:15pm", "9am")


@dataclass(frozen=True)
class ToolRequest:
    text: str
    tool: str
    slots: dict
    split: str


def tool_requests(n: int, *, split: str, seed: int, order_ids: list[str]) -> list[ToolRequest]:
    rng = random.Random(f"tools:{split}:{seed}")
    out = []
    for i in range(n):
        tool = TOOLS[i % len(TOOLS)]
        slots = {"o": rng.choice(order_ids), "d": rng.choice(DAY_PHRASES), "t": rng.choice(TIME_PHRASES),
                 "a": rng.choice(STREETS)}
        text = rng.choice(TOOL_TEMPLATES[tool][split]).format(**slots)
        used = {k: v for k, v in slots.items() if "{" + k + "}" in " ".join(TOOL_TEMPLATES[tool][split])}
        out.append(ToolRequest(text=_join(rng.choice(OPENERS), text), tool=tool, slots=used, split=split))
    return out


# --------------------------------------------------------------------------- reply pairs

@dataclass(frozen=True)
class ReplyPair:
    customer: str
    order_id: str
    issue: str
    reply_a: str
    reply_b: str
    better: str  # "A" or "B"
    defect: str  # what is wrong with the worse reply


GOOD_REPLY = ("Hi {first}, thank you for letting us know about order {o}. I am sorry the {p} {issue_past}. "
              "I have opened a case and you will hear from us within 24 hours with the next step.")
DEFECTS = {
    "missing_order": ("Hi {first}, thank you for letting us know. I am sorry the {p} {issue_past}. "
                      "I have opened a case and you will hear from us within 24 hours with the next step."),
    "wrong_name": ("Hi {wrong}, thank you for letting us know about order {o}. I am sorry the {p} {issue_past}. "
                   "I have opened a case and you will hear from us within 24 hours with the next step."),
    "rude": ("{first}, order {o} left our warehouse in perfect condition, so whatever happened to the {p} is not "
             "really our problem. You can open a case if you insist."),
    "no_next_step": ("Hi {first}, about order {o}: sorry about the {p}. These things happen sometimes."),
    "false_promise": ("Hi {first}, thank you for letting us know about order {o}. I am sorry the {p} {issue_past}. "
                      "I have refunded you twice the amount and added a free gift to your account."),
}
ISSUES = {"damaged": "arrived damaged", "not_received": "has not arrived yet", "wrong_item": "was not what you ordered"}


def reply_pairs(n: int, *, seed: int, world: World) -> list[ReplyPair]:
    rng = random.Random(f"pairs:{seed}")
    delivered = world.orders[world.orders.status == "delivered"].reset_index(drop=True)
    out = []
    defects = tuple(DEFECTS)
    for i in range(n):
        row = delivered.iloc[rng.randrange(len(delivered))]
        cust = world.customer(row.customer_id)
        first = cust["name"].split()[0]
        wrong = rng.choice([f for f in FIRST if f != first])
        issue = rng.choice(tuple(ISSUES))
        fmt = {"first": first, "wrong": wrong, "o": row.order_id, "p": row["product"], "issue_past": ISSUES[issue]}
        good = GOOD_REPLY.format(**fmt)
        defect = defects[i % len(defects)]
        bad = DEFECTS[defect].format(**fmt)
        better = "A" if rng.random() < 0.5 else "B"
        a, b = (good, bad) if better == "A" else (bad, good)
        out.append(ReplyPair(customer=row.customer_id, order_id=row.order_id, issue=issue, reply_a=a, reply_b=b,
                             better=better, defect=defect))
    return out


# --------------------------------------------------------------------------- cost estimates

ESTIMATE_TEMPLATES = {
    "train": (("If we refund {p}% of {n} orders that average ${a}, what is the total refund cost?",
               lambda p, n, a: p / 100 * n * a),
              ("What does it cost to refund {p} percent of {n} orders at ${a} each?",
               lambda p, n, a: p / 100 * n * a)),
    "test": (("Suppose {n} orders, average basket ${a}, and a {p}% refund rate. How many dollars go back to customers?",
              lambda p, n, a: p / 100 * n * a),
             ("We ship {n} parcels at ${a} per label and {p}% need a second label. What is the total label spend?",
              lambda p, n, a: n * a + p / 100 * n * a)),
}


@dataclass(frozen=True)
class Estimate:
    text: str
    answer: float
    split: str


def estimates(n: int, *, split: str, seed: int) -> list[Estimate]:
    rng = random.Random(f"est:{split}:{seed}")
    out = []
    for i in range(n):
        tmpl, fn = ESTIMATE_TEMPLATES[split][i % len(ESTIMATE_TEMPLATES[split])]
        p, n_, a = rng.choice((5, 8, 10, 12, 15, 20)), rng.choice((120, 250, 340, 500, 800)), rng.choice((12, 25, 45, 60, 80))
        out.append(Estimate(text=tmpl.format(p=p, n=n_, a=a), answer=round(fn(p, n_, a), 2), split=split))
    return out


# --------------------------------------------------------------------------- policy questions

POLICY_QUERIES = {
    "P1": {"train": ("changing my mind about a purchase", "the return window", "returning an item I no longer want"),
           "test": ("how long I have to send back a sweater I just do not like", "regret purchases")},
    "P2": {"train": ("damaged items", "broken or defective products", "items that arrive broken"),
           "test": ("a lamp that showed up cracked", "goods that do not work out of the box")},
    "P3": {"train": ("orders that never arrived", "items not received", "missing deliveries"),
           "test": ("a parcel that is two weeks late", "packages lost by the courier")},
    "P4": {"train": ("receiving the wrong item", "wrong size or colour", "getting someone else's product"),
           "test": ("a jacket in the wrong size", "an order mixed up with another one")},
    "P5": {"train": ("expense approval", "expense reports without a receipt", "expenses over 500 dollars"),
           "test": ("claiming a client dinner", "reimbursement for a taxi ride")},
    "P6": {"train": ("phone callbacks", "scheduling a call", "callback hours"),
           "test": ("talking to an agent on the phone", "when someone can ring me")},
    "P7": {"train": ("changing my shipping address", "address changes", "updating the delivery address"),
           "test": ("moving house before my order arrives", "sending a parcel to my office instead")},
    "P8": {"train": ("cancelling an order", "order cancellations", "stopping an order before it ships"),
           "test": ("calling off a purchase I made by accident", "undoing an order")},
}


@dataclass(frozen=True)
class PolicyQuery:
    text: str
    section: str
    split: str


def policy_queries(n: int, *, split: str, seed: int) -> list[PolicyQuery]:
    rng = random.Random(f"pq:{split}:{seed}")
    ids = sorted(POLICY_QUERIES)
    return [PolicyQuery(text=rng.choice(POLICY_QUERIES[s][split]), section=s, split=split)
            for s in (ids[i % len(ids)] for i in range(n))]
