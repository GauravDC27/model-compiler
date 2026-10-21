from dmc.compiler import Clarify, Compiler, RuleParser, lock_hash_for, normalize
from dmc.ir import Context
from dmc.profiles.decide import SYSTEM, DecideProfile, label_orderings, render_user
from dmc.runtime import TableBackend
from dmc.store import ResultStore

CTX = Context(now="2026-10-21T00:00:00Z", locale="en-US", timezone="UTC")


def backend_for(question, yes, no):
    """A fake model that prefers `yes`/`no` by the given log-probs, in every label order."""
    from dmc.ir import DecideClassify

    op = DecideClassify(context=CTX, ambiguities=(), question=question, labels=("yes", "no"),
                        policy=None, evidence=())
    b = TableBackend({})
    for order in label_orderings(op.labels):
        prompt = b.render_chat(SYSTEM, render_user(op, order))
        b.scores[(prompt, "yes")] = yes
        b.scores[(prompt, "no")] = no
    return b


def make(tmp_path, backend):
    return Compiler(RuleParser(), {"decide": DecideProfile(backend)}, ResultStore(tmp_path),
                    lock_hash_for(backend.lock_fields(), RuleParser.version))


def test_normalize_collapses_surface_variation():
    assert normalize("Is  91 prime?’ ") == normalize("Is 91 prime?'")


def test_decision_then_replay(tmp_path):
    c = make(tmp_path, backend_for("Is 91 prime?", yes=-3.0, no=-0.1))
    first = c.run("Is 91 prime? Answer yes or no.", CTX)
    assert first.result.label == "no" and not first.replayed
    again = c.run("Is   91 prime?  Answer yes or no.", CTX)  # equal after normalization
    assert again.replayed and again.result == first.result
    assert again.receipt.key == first.receipt.key


def test_low_margin_abstains(tmp_path):
    c = make(tmp_path, backend_for("Approve?", yes=-0.70, no=-0.69))
    out = c.run("Approve? Answer yes or no.", CTX)
    assert out.result.status == "abstained" and out.result.label is None


def test_out_of_grammar_is_clarify_not_guess(tmp_path):
    c = make(tmp_path, backend_for("x", 0, 0))
    out = c.run("Write me a poem about the sea", CTX)
    assert isinstance(out.result, Clarify)


def test_case_marginalization_reads_capitalized_preference(tmp_path):
    from dmc.ir import DecideClassify

    op = DecideClassify(context=CTX, ambiguities=(), question="Is 91 prime?", labels=("yes", "no"),
                        policy=None, evidence=())
    b = TableBackend({})
    for order in label_orderings(op.labels):
        prompt = b.render_chat(SYSTEM, render_user(op, order))
        # The model says "No" (capitalized) with high probability, lowercase forms are rare.
        b.scores.update({(prompt, "yes"): -4.0, (prompt, "no"): -4.2, (prompt, "No"): -0.1,
                         (prompt, "Yes"): -3.0})
    plain = DecideProfile(b, abstain_margin=0.0).run(op)
    marg = DecideProfile(b, abstain_margin=0.0, marginalize_case=True).run(op)
    assert plain.label == "yes"  # misled by lowercase-only scoring
    assert marg.label == "no"
