"""Part 1: meaning as geometry inside a pretrained embedding table.

The potion-base-8M table (Model2Vec) holds one 256-dim row per token of a
29,528-word vocabulary.  A relation between words (country -> capital, ...)
shows up as a direction: the offsets of pairs that share the relation point
the same way.  The shuffled-row control keeps every vector but hands each
word another word's vector, so any pattern that survives the shuffle comes
from the vectors alone, not from what the words mean.
"""

from __future__ import annotations

import json
import os

import numpy as np

WEEK8_DIR = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(WEEK8_DIR, "inputs", "potion-base-8M", "model.safetensors")
VOCAB_PATH = os.path.join(WEEK8_DIR, "inputs", "potion-base-8M", "tokenizer.json")

# The nine word pairs for each of the three relations, verbatim from the
# week 8 learning sheet (VERIFY 1).
RELATIONS: dict[str, list[tuple[str, str]]] = {
    "country to capital": [
        ("iran", "tehran"), ("turkey", "ankara"), ("russia", "moscow"),
        ("china", "beijing"), ("france", "paris"), ("sweden", "stockholm"),
        ("japan", "tokyo"), ("peru", "lima"), ("kenya", "nairobi"),
    ],
    "male to female": [
        ("king", "queen"), ("man", "woman"), ("he", "she"),
        ("brother", "sister"), ("father", "mother"), ("son", "daughter"),
        ("uncle", "aunt"), ("boy", "girl"), ("husband", "wife"),
    ],
    "adjective to comparative": [
        ("big", "bigger"), ("small", "smaller"), ("fast", "faster"),
        ("dark", "darker"), ("cold", "colder"), ("old", "older"),
        ("strong", "stronger"), ("long", "longer"), ("tall", "taller"),
    ],
}


def load_table(path: str = TABLE_PATH) -> np.ndarray:
    """Load the embedding table without PyTorch: safetensors -> numpy."""
    from safetensors.numpy import load_file

    tensors = load_file(path)
    assert len(tensors) == 1, tensors.keys()
    E = next(iter(tensors.values())).astype(np.float64)
    assert E.ndim == 2
    return E


def load_vocab(path: str = VOCAB_PATH) -> dict[str, int]:
    """Word -> token id, parsed straight from tokenizer.json."""
    with open(path) as f:
        vocab = json.load(f)["model"]["vocab"]
    return {w: int(i) for w, i in vocab.items()}


def unit_rows(E: np.ndarray) -> np.ndarray:
    """Scale every row to length one."""
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def shuffled_control(E: np.ndarray, seed: int = 1730) -> np.ndarray:
    """A copy of the table with its rows shuffled: same vectors, wrong words."""
    perm = np.random.default_rng(seed).permutation(E.shape[0])
    return E[perm]


def pair_offsets(E_hat: np.ndarray, pairs, vocab: dict[str, int]) -> np.ndarray:
    """Offsets E_hat[b] - E_hat[a] of the pairs, rows already unit length."""
    return np.stack([E_hat[vocab[b]] - E_hat[vocab[a]] for a, b in pairs])


def leave_one_out_cosines(E_hat: np.ndarray, pairs, vocab: dict[str, int]) -> np.ndarray:
    """Cosine of each pair's offset with the mean offset of the other eight.

    Leaving the pair out of the average keeps it from agreeing with itself.
    """
    offsets = pair_offsets(E_hat, pairs, vocab)
    n = len(pairs)
    total = offsets.sum(axis=0)
    others = (total[None, :] - offsets) / (n - 1)      # mean of the other 8
    num = np.sum(offsets * others, axis=1)
    den = np.linalg.norm(offsets, axis=1) * np.linalg.norm(others, axis=1)
    return num / den
