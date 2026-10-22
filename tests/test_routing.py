import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dmc.routing import (  # noqa: E402
    TOKENS,
    RoutePinner,
    RouteTracer,
    TraceMismatch,
    TraceOverrun,
    find_routers,
    harden_routers,
    trace_size_bytes,
)


def tiny(kind: str, dtype=torch.float32):
    torch.manual_seed(0)
    if kind == "granite":
        cfg = transformers.GraniteMoeConfig(
            vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=6, num_experts_per_tok=2,
        )
        return transformers.GraniteMoeForCausalLM(cfg).to(dtype).eval()
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=6,
        num_experts_per_tok=2, head_dim=8, norm_topk_prob=True,
    )
    return transformers.Qwen3MoeForCausalLM(cfg).to(dtype).eval()


@pytest.mark.parametrize("kind", ["granite", "qwen3"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pinning_own_trace_is_bitwise_identity(kind, dtype):
    model = tiny(kind, dtype)
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad(), RouteTracer(model) as tr:
        ref = model(input_ids=ids).logits
    with torch.no_grad(), RoutePinner(model, tr.trace()) as pin:
        out = model(input_ids=ids).logits
    assert pin.status == "full"
    assert torch.equal(ref, out)


@pytest.mark.parametrize("kind", ["granite", "qwen3"])
def test_pinning_forces_recorded_routes_on_a_perturbed_model(kind):
    model = tiny(kind)
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad(), RouteTracer(model) as tr:
        model(input_ids=ids)
    ref_trace = tr.trace()
    with torch.no_grad():
        for _, mod, _ in find_routers(model):
            mod.weight.add_(torch.randn_like(mod.weight))
    with torch.no_grad(), RouteTracer(model) as live:
        model(input_ids=ids)
    assert any(not torch.equal(live.trace()[k], ref_trace[k]) for k in ref_trace if k != TOKENS)
    # The tracer wraps the pinner, so it observes the pinned indices.
    with torch.no_grad(), RoutePinner(model, ref_trace), RouteTracer(model) as seen:
        model(input_ids=ids)
    for k in ref_trace:
        assert torch.equal(seen.trace()[k], ref_trace[k])


def test_generation_trace_replays_prefill_and_decode():
    model = tiny("granite")
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad(), RouteTracer(model) as tr:
        ref = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False, pad_token_id=0)
    with torch.no_grad(), RoutePinner(model, tr.trace()) as pin:
        out = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False, pad_token_id=0)
    assert pin.status == "full"
    assert torch.equal(ref, out)


def test_teacher_forced_replay_of_a_generation_trace():
    model = tiny("granite")
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad(), RouteTracer(model) as tr:
        model.generate(input_ids=ids, max_new_tokens=5, do_sample=False, pad_token_id=0)
    trace = tr.trace()
    fed = trace[TOKENS][None]  # prompt + every generated token that was fed back
    with torch.no_grad(), RoutePinner(model, trace) as pin:
        model(input_ids=fed)
    assert pin.status == "full"


def test_strict_pinner_rejects_other_tokens_and_overrun():
    model = tiny("granite")
    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad(), RouteTracer(model) as tr:
        model(input_ids=ids)
    other = (ids + 1) % 64
    with pytest.raises(TraceMismatch), torch.no_grad(), RoutePinner(model, tr.trace()):
        model(input_ids=other)
    longer = torch.cat([ids, ids], dim=1)
    with pytest.raises((TraceOverrun, TraceMismatch)), torch.no_grad(), RoutePinner(model, tr.trace()):
        model(input_ids=longer)
    with torch.no_grad(), RoutePinner(model, tr.trace(), strict=False) as lax:
        model(input_ids=other)
    assert lax.status == "partial"


def test_trace_size_excludes_tokens():
    model = tiny("granite")
    with torch.no_grad(), RouteTracer(model) as tr:
        model(input_ids=torch.randint(0, 64, (1, 10)))
    assert trace_size_bytes(tr.trace()) == 2 * 10 * 2  # layers * tokens * top_k


@pytest.mark.parametrize("kind", ["granite", "qwen3"])
def test_hardened_router_matches_stock_routes_in_fp32_and_breaks_ties_low(kind):
    model = tiny(kind)
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad(), RouteTracer(model) as stock:
        ref = model(input_ids=ids).logits
    with torch.no_grad(), harden_routers(model), RouteTracer(model) as hard:
        out = model(input_ids=ids).logits
    for k in stock.trace():
        assert torch.equal(stock.trace()[k].sort(-1).values, hard.trace()[k].sort(-1).values)
    assert torch.allclose(ref, out, atol=1e-5)
    # Exact ties: all-zero router weights make every logit equal; lowest indices win.
    with torch.no_grad():
        for _, mod, _ in find_routers(model):
            mod.weight.zero_()
    with torch.no_grad(), harden_routers(model), RouteTracer(model) as tied:
        model(input_ids=ids)
    for k, v in tied.trace().items():
        if k != TOKENS:
            assert torch.equal(v, torch.tensor([[0, 1]]).expand_as(v))


def test_fragile_share_can_skip_a_shared_prefix():
    model = tiny("granite")
    with torch.no_grad(), RouteTracer(model, record_margins=True) as tr:
        model(input_ids=torch.randint(0, 64, (1, 10)))
    full = tr.fragile_share(1e9)
    assert full == 1.0
    assert tr.fragile_share(1e9, skip=4) == 1.0
    assert tr.fragile_share(-1.0, skip=4) == 0.0
    assert tr.fragile_share(1e9, skip=10) == 0.0  # nothing left after the prefix
