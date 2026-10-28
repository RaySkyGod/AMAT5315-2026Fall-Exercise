import numpy as np
import pytest

from embeddings import (
    RELATIONS,
    leave_one_out_cosines,
    load_table,
    load_vocab,
    shuffled_control,
    unit_rows,
)


def test_unit_rows_scales_each_row_to_length_one():
    E = np.random.default_rng(0).normal(size=(5, 8))
    U = unit_rows(E)
    assert U.shape == E.shape
    assert np.allclose(np.linalg.norm(U, axis=1), 1.0)


def test_control_is_row_permutation_and_reproducible():
    E = np.arange(20, dtype=float).reshape(4, 5)
    C1 = shuffled_control(E, 1730)
    C2 = shuffled_control(E, 1730)
    assert np.array_equal(C1, C2)                      # same seed, same result
    assert sorted(C1.ravel().tolist()) == sorted(E.ravel().tolist())
    assert not np.array_equal(C1, E)                   # rows actually moved


def test_loo_cosine_is_one_when_offsets_are_parallel():
    rng = np.random.default_rng(1)
    d = 16
    u = np.zeros(d)
    u[0] = 1.0                                          # the shared direction
    E = np.zeros((18, d))
    pairs = []
    key = {}
    for i in range(9):
        a = rng.normal(size=d)
        E[2 * i] = a
        E[2 * i + 1] = a + u                           # offset exactly +u
        key[f"a{i}"] = 2 * i
        key[f"b{i}"] = 2 * i + 1
        pairs.append((f"a{i}", f"b{i}"))
    cos = leave_one_out_cosines(E, pairs, key)
    assert cos.shape == (9,)
    assert np.allclose(cos, 1.0, atol=1e-12)           # every pair agrees


def test_loo_cosine_near_zero_for_shuffled_rows():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(18, 256))
    E = unit_rows(E)
    key = {f"w{i}": i for i in range(18)}
    pairs = [(f"w{2*i}", f"w{2*i+1}") for i in range(9)]
    C = shuffled_control(E, seed=1730)
    cos = leave_one_out_cosines(C, pairs, key)
    assert np.abs(cos).max() < 0.5                     # unrelated directions


def test_relations_have_nine_pairs_each():
    assert set(RELATIONS) == {
        "country to capital",
        "male to female",
        "adjective to comparative",
    }
    for pairs in RELATIONS.values():
        assert len(pairs) == 9
        for a, b in pairs:
            assert isinstance(a, str) and isinstance(b, str)


@pytest.mark.skipif(
    not __import__("os").path.exists("inputs/potion-base-8M/model.safetensors"),
    reason="potion table not downloaded",
)
def test_table_and_vocab_load():
    E = load_table("inputs/potion-base-8M/model.safetensors")
    vocab = load_vocab("inputs/potion-base-8M/tokenizer.json")
    assert E.shape == (29528, 256)
    assert vocab["cat"] == 3943 and vocab["kitten"] == 17407
    for pairs in RELATIONS.values():
        for a, b in pairs:
            assert a in vocab and b in vocab, (a, b)
