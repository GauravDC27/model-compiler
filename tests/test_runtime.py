import pytest

from dmc.runtime import FIXED_EPOCH, position_seed, template_clock


def test_position_seed_is_a_pure_function():
    assert position_seed(7, 3) == position_seed(7, 3)
    assert len({position_seed(7, p) for p in range(100)}) == 100


def test_template_clock_binds_to_context_or_fixed_epoch():
    assert template_clock("2020-01-02T00:00:00Z")["strftime_now"]("%Y-%m-%d") == "2020-01-02"
    assert template_clock("2020-01-02T00:00:00Z")["date_string"] == "02 Jan 2020"
    assert template_clock(None)["strftime_now"]("%Y") == str(FIXED_EPOCH.year)


def test_chat_template_cannot_read_the_host_clock():
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    tok.chat_template = "{{ strftime_now('%Y-%m-%d') }}|{{ date_string }}|{% for m in messages %}{{ m.content }}{% endfor %}"
    msgs = [{"role": "user", "content": "hi"}]
    a = tok.apply_chat_template(msgs, tokenize=False, **template_clock("2020-01-02T00:00:00Z"))
    b = tok.apply_chat_template(msgs, tokenize=False, **template_clock(None))
    assert a == "2020-01-02|02 Jan 2020|hi"
    assert b == "1970-01-01|01 Jan 1970|hi"
