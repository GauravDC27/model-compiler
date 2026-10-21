"""dmc-IR: the typed intermediate representation every request is compiled to.

The IR is the "source code" of the model compiler. It is a closed discriminated union of
ops whose slots are all required (nullable rather than optional), so two requests with the
same meaning serialize to the same canonical bytes and the same hash.

An IR with a non-empty ``ambiguities`` list is ill-formed: it must be answered with a
``Clarify`` result instead of being executed (defined failure, contract level D-bottom).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .canonical import canonical_json, content_hash

IR_SCHEMA_VERSION = "dmc.ir/0"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Context(_Frozen):
    """Everything ambient made explicit: time and locale are inputs, never read from the host."""

    now: str | None  # ISO-8601 timestamp supplied by the caller
    locale: str | None
    timezone: str | None


class Ambiguity(_Frozen):
    slot: str
    question: str
    candidates: tuple[str, ...]


class _Op(_Frozen):
    context: Context
    ambiguities: tuple[Ambiguity, ...]

    @property
    def executable(self) -> bool:
        return not self.ambiguities


class DecideClassify(_Op):
    op: Literal["decide.classify"] = "decide.classify"
    question: str
    labels: tuple[str, ...]
    policy: str | None
    evidence: tuple[str, ...]

    @field_validator("labels")
    @classmethod
    def _labels_closed_and_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(v) < 2:
            raise ValueError("a decision needs at least two labels")
        if len(set(v)) != len(v):
            raise ValueError("labels must be unique")
        return v


class ToolCall(_Op):
    op: Literal["tool.call"] = "tool.call"
    tool: str | None  # None when the tool must be selected from `allowed_tools`
    allowed_tools: tuple[str, ...]
    arguments: tuple[tuple[str, str | int | float | bool | None], ...]
    instruction: str


class EvaluatePairwise(_Op):
    op: Literal["evaluate.pairwise"] = "evaluate.pairwise"
    criterion: str
    candidate_a: str
    candidate_b: str


class EvaluateRubric(_Op):
    op: Literal["evaluate.rubric"] = "evaluate.rubric"
    criterion: str
    candidate: str
    scale: tuple[int, int]


class CreateText(_Op):
    op: Literal["create.text"] = "create.text"
    brief: str
    form: str  # e.g. "poem", "email", "story"
    constraints: tuple[str, ...]
    seed: int | None  # None => derived from the IR hash (same prompt, same output)


class AnalyzeReason(_Op):
    op: Literal["analyze.reason"] = "analyze.reason"
    question: str
    answer_format: Literal["number", "label", "text", "json"]
    sources: tuple[str, ...]  # content hashes of snapshotted inputs


Op = Annotated[
    Union[DecideClassify, ToolCall, EvaluatePairwise, EvaluateRubric, CreateText, AnalyzeReason],
    Field(discriminator="op"),
]

_OP_ADAPTER: TypeAdapter[Op] = TypeAdapter(Op)


class Program(_Frozen):
    """A compiled request: one or more ops plus the schema version they were compiled for."""

    schema_version: Literal["dmc.ir/0"] = IR_SCHEMA_VERSION
    ops: tuple[Op, ...]

    @property
    def executable(self) -> bool:
        return all(op.executable for op in self.ops)

    def canonical(self) -> str:
        return canonical_json(self)

    @property
    def hash(self) -> str:
        return content_hash(self)


def parse_op(data: dict) -> Op:
    return _OP_ADAPTER.validate_python(data)


def op_json_schema() -> dict:
    """JSON Schema of the op union: the grammar the NL->IR parser is constrained to."""
    return _OP_ADAPTER.json_schema()
