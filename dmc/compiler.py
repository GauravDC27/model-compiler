"""The compiler pipeline: normalize -> parse -> analyze -> canonicalize -> dispatch -> execute
-> verify -> store, with a receipt on every result (plan section 3).

Phase-0 scope: the pipeline, the static dispatch table, the result store and the Decide
profile are real; the NL->IR parser is pluggable (a narrow deterministic rule parser ships
here; the constrained-decoding LLM parser is Phase 2).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from .canonical import canonical_json, content_hash
from .ir import Ambiguity, Context, DecideClassify, Program
from .store import Receipt, Record, ResultStore, result_key


def normalize(text: str) -> str:
    """Deterministic surface normalization: NFC, unified quotes, collapsed whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


class Clarify(BaseModel):
    """Defined failure: the request is ambiguous or outside the IR; ask instead of guessing."""

    status: str = "clarify"
    reason: str
    ambiguities: tuple[Ambiguity, ...] = ()


class Parser(Protocol):
    version: str

    def parse(self, text: str, context: Context) -> Program | Clarify: ...


_YES_NO = re.compile(
    r"^(?P<q>.+?)\s*(?:answer|reply|respond)(?: with| only)? (?:yes or no|with yes or no)\.?$",
    re.IGNORECASE,
)


class RuleParser:
    """A narrow, fully deterministic parser for yes/no decision requests.

    It demonstrates the contract, not coverage: anything it does not recognize is a
    ``Clarify`` (out of grammar), never a guess.
    """

    version = "rule.p0"

    def parse(self, text: str, context: Context) -> Program | Clarify:
        m = _YES_NO.match(text)
        if not m:
            return Clarify(reason="request is outside the grammar of parser rule.p0")
        question = m.group("q").strip()
        # Context minimization: a yes/no question does not depend on time or time zone.
        ctx = Context(now=None, locale=context.locale, timezone=None,
                      session_state_hash=context.session_state_hash)
        op = DecideClassify(
            context=ctx, ambiguities=(), question=question, labels=("yes", "no"),
            policy=None, evidence=(),
        )
        return Program(ops=(op,))


# Versioned static dispatch table (plan §3 step 5). Changing it is a lock change.
DISPATCH_TABLE: dict[str, str] = {
    "decide": "decide",
    "tool": "call",
    "evaluate": "judge",
    "create": "create",
    "analyze": "reason",
    "research": "reason",
}


class Profile(Protocol):
    name: str
    version: str
    result_type: type[BaseModel]

    def run(self, op) -> BaseModel: ...


@dataclass(frozen=True)
class Compiled:
    result: BaseModel
    receipt: Receipt | None
    replayed: bool


class Compiler:
    def __init__(
        self,
        parser: Parser,
        profiles: dict[str, Profile],
        store: ResultStore,
        lock_hash: str,
        verifiers: dict[str, Callable[[object, BaseModel], dict[str, str]]] | None = None,
    ) -> None:
        self.parser = parser
        self.profiles = profiles
        self.store = store
        self.lock_hash = lock_hash
        self.verifiers = verifiers or {}

    def dispatch(self, op) -> Profile:
        """Static dispatch: IR op family -> profile name (DISPATCH_TABLE) -> registered profile."""
        family = op.op.split(".", 1)[0]
        name = DISPATCH_TABLE.get(family)
        if name is None or name not in self.profiles:
            raise KeyError(f"no profile registered for op family {family!r}")
        return self.profiles[name]

    def compile(self, prompt: str, context: Context) -> Program | Clarify:
        return self.parser.parse(normalize(prompt), context)

    def run(self, prompt: str, context: Context) -> Compiled:
        program = self.compile(prompt, context)
        if isinstance(program, Clarify):
            return Compiled(program, None, False)
        if not program.executable:
            amb = tuple(a for op in program.ops for a in op.ambiguities)
            return Compiled(Clarify(reason="ambiguous request", ambiguities=amb), None, False)
        if len(program.ops) != 1:
            raise NotImplementedError("multi-op programs run on the durable executor (Phase 3)")
        op = program.ops[0]
        profile = self.dispatch(op)
        seed = getattr(op, "seed", None)
        key = result_key(self.lock_hash, program.hash, f"{profile.name}:{profile.version}", seed)
        hit = self.store.get(key, lock_hash=self.lock_hash)
        if hit is not None:
            return Compiled(profile.result_type.model_validate_json(hit.output), hit.receipt, True)
        result = profile.run(op)
        verdict = self.verifiers.get(profile.name, lambda _op, _r: {})(op, result)
        margins = {"decision": result.margin} if hasattr(result, "margin") else {}
        receipt = Receipt(
            key=key, lock_hash=self.lock_hash, ir_hash=program.hash,
            session_state_hash=op.context.session_state_hash,
            rendered_prompt_hash=getattr(result, "rendered_prompt_hash", None), profile=profile.name,
            profile_version=profile.version, seed=seed, margins=margins,
            coverage_band=getattr(profile, "abstain_margin", None), verifier=verdict,
        )
        self.store.put(Record(receipt=receipt, output=canonical_json(result)))
        return Compiled(result, receipt, False)


def lock_hash_for(fields: dict[str, object], parser_version: str) -> str:
    return content_hash({"runtime": fields, "parser": parser_version})
