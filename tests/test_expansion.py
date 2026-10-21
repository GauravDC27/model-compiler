import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from dmc.expansion import expand, init_goal_bias, set_goal, trainable_parameters  # noqa: E402


def tiny():
    torch.manual_seed(0)
    cfg = transformers.GraniteMoeConfig(
        vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, num_local_experts=6, num_experts_per_tok=2,
    )
    return transformers.GraniteMoeForCausalLM(cfg).eval()


@pytest.mark.parametrize("variant", ["routed", "additive"])
def test_goal_off_and_unknown_goal_are_bit_identical_to_base(variant):
    model = tiny()
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    wrapped = expand(model, [0, 1], n_new=2, goals=["math"], variant=variant)
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, ref)
        set_goal(wrapped, "poetry")  # a goal these experts were not built for
        assert torch.equal(model(input_ids=ids).logits, ref)


def test_zero_initialized_new_experts_are_a_no_op_even_when_selected():
    model = tiny()
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    wrapped = expand(model, [0, 1], n_new=1, goals=["math"], variant="additive")
    set_goal(wrapped, "math")
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, ref, atol=1e-6)


def test_routed_goal_bias_sends_traffic_to_new_experts_and_only_new_params_learn():
    model = tiny()
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = expand(model, [0, 1], n_new=1, goals=["math"], variant="routed")
    set_goal(wrapped, "math")
    init_goal_bias(wrapped, "math", 100.0)  # force selection
    ids = torch.randint(0, 64, (2, 8))
    h = model.model.embed_tokens(ids)
    assert wrapped[0].new_expert_share(h) == 1.0
    params = trainable_parameters(wrapped)
    for p in params:
        p.requires_grad_(True)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    assert all(p.grad is not None for p in params)
    assert any(float(p.grad.abs().sum()) > 0 for p in params)
    assert all(p.grad is None for p in model.parameters() if not any(p is q for q in params))
    set_goal(wrapped, None)
    assert wrapped[0].new_expert_share(h) == 0.0


def test_parent_router_init_with_noise_keeps_goal_off_identity_and_starts_near_parent():
    model = tiny()
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    parent_row = model.model.layers[0].block_sparse_moe.router.weight[3].detach().clone()
    g = torch.Generator().manual_seed(0)
    wrapped = expand(model, [0, 1], 1, ["math"], "routed", {0: [3], 1: [3]},
                     noise=0.05, router_init="parent", bias_init=0.1, generator=g)
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, ref)  # goal off: base untouched
    row = wrapped[0].router_new.detach()[0]
    assert not torch.equal(row, parent_row)
    assert float((row - parent_row).norm() / parent_row.norm()) < 0.2
    assert float(wrapped[0].goal_bias[0, 0]) == pytest.approx(0.1)


def test_hot_expert_lora_starts_as_identity_and_trains_only_its_parameters():
    from dmc.expert_lora import attach, detach

    model = tiny()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    mods = attach(model, {0: [0, 2], 1: [1]}, rank=2, alpha=4.0)
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, ref, atol=1e-5)
    params = [p for m in mods for p in (m.a_gu, m.b_gu, m.a_d, m.b_d)]
    for p in params:
        p.requires_grad_(True)
    model(input_ids=ids, labels=ids).loss.backward()
    assert any(float(p.grad.abs().sum()) > 0 for p in params)
    detach(model, mods, [0, 1])
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, ref)
