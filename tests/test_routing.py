import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dmc.routing import RoutePinner, RouteTracer, find_routers  # noqa: E402


def tiny(kind: str):
    torch.manual_seed(0)
    if kind == "granite":
        cfg = transformers.GraniteMoeConfig(
            vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, num_local_experts=6, num_experts_per_tok=2,
        )
        return transformers.GraniteMoeForCausalLM(cfg).eval()
    cfg = transformers.Qwen3MoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=32, moe_intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, num_experts=6,
        num_experts_per_tok=2, head_dim=8, norm_topk_prob=True,
    )
    return transformers.Qwen3MoeForCausalLM(cfg).eval()


@pytest.mark.parametrize("kind", ["granite", "qwen3"])
def test_pinning_own_trace_is_bitwise_identity(kind):
    model = tiny(kind)
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad(), RouteTracer(model) as tr:
        ref = model(input_ids=ids).logits
    with torch.no_grad(), RoutePinner(model, tr.trace()) as pin:
        out = model(input_ids=ids).logits
    assert pin.overruns == 0
    assert torch.equal(ref, out)


@pytest.mark.parametrize("kind", ["granite", "qwen3"])
def test_pinning_forces_recorded_routes_on_a_perturbed_model(kind):
    model = tiny(kind)
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad(), RouteTracer(model) as tr:
        model(input_ids=ids)
    ref_trace = tr.trace()
    # Perturb the routers so live routing differs from the recorded routes.
    with torch.no_grad():
        for _, mod, _ in find_routers(model):
            mod.weight.add_(torch.randn_like(mod.weight))
    with torch.no_grad(), RouteTracer(model) as live:
        model(input_ids=ids)
    assert any(not torch.equal(live.trace()[k], ref_trace[k]) for k in ref_trace)
    # Tracer outside, pinner inside: the tracer sees the pinned indices.
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
    assert pin.overruns == 0
    assert torch.equal(ref, out)
