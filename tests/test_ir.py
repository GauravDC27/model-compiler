import pytest
from pydantic import ValidationError

from dmc.ir import Ambiguity, Context, DecideClassify, Program, op_json_schema, parse_op

CTX = Context(now="2026-10-21T00:00:00Z", locale="en-US", timezone="UTC")


def decide(**kw):
    base = dict(context=CTX, ambiguities=(), question="Approve?", labels=("yes", "no"),
                policy=None, evidence=("score=580",))
    base.update(kw)
    return DecideClassify(**base)


def test_equal_meaning_equal_hash():
    a = Program(ops=(decide(),))
    b = Program.model_validate_json(a.model_dump_json())
    assert a.hash == b.hash


def test_slots_are_required():
    with pytest.raises(ValidationError):
        DecideClassify(context=CTX, ambiguities=(), question="q", labels=("y", "n"), evidence=())


def test_labels_must_be_closed_and_unique():
    with pytest.raises(ValidationError):
        decide(labels=("yes",))
    with pytest.raises(ValidationError):
        decide(labels=("yes", "yes"))


def test_ambiguity_makes_program_non_executable():
    amb = Ambiguity(slot="evidence", question="Which applicant?", candidates=("A", "B"))
    assert not Program(ops=(decide(ambiguities=(amb,)),)).executable
    assert Program(ops=(decide(),)).executable


def test_discriminated_union_round_trip_and_schema():
    op = parse_op(decide().model_dump(mode="json"))
    assert isinstance(op, DecideClassify)
    schema = op_json_schema()
    assert "discriminator" in schema or "oneOf" in schema


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        parse_op({**decide().model_dump(mode="json"), "temperature": 0.7})
