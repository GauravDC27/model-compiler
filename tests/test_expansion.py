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
