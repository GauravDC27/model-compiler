import pytest

from dmc.store import NondeterminismError, Receipt, Record, ResultStore, result_key


def rec(key, lock, out):
    return Record(receipt=Receipt(key=key, lock_hash=lock, ir_hash="sha256:ir", profile="decide",
                                  profile_version="1", seed=None), output=out)


def test_replay_and_lock_mismatch(tmp_path):
    store = ResultStore(tmp_path)
    key = result_key("sha256:lock", "sha256:ir", "1", None)
    store.put(rec(key, "sha256:lock", '{"label":"no"}'))
    assert store.get(key, lock_hash="sha256:lock").output == '{"label":"no"}'
    assert store.get(key, lock_hash="sha256:other") is None


def test_divergent_recompute_is_an_error(tmp_path):
    store = ResultStore(tmp_path)
    key = result_key("sha256:lock", "sha256:ir", "1", None)
    store.put(rec(key, "sha256:lock", '{"label":"no"}'))
    store.put(rec(key, "sha256:lock", '{"label":"no"}'))  # identical recompute is fine
    with pytest.raises(NondeterminismError):
        store.put(rec(key, "sha256:lock", '{"label":"yes"}'))


def test_key_depends_on_every_component():
    base = result_key("l", "i", "p", 1)
    assert len({base, result_key("l2", "i", "p", 1), result_key("l", "i2", "p", 1),
                result_key("l", "i", "p2", 1), result_key("l", "i", "p", 2)}) == 5
