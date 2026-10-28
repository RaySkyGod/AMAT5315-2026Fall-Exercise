"""Part 2: the TinyStories corpus, its split, and the BPE token streams.

The validation file of TinyStoriesV2-GPT4 holds 27,630 stories separated by
`<|endoftext|>` markers.  We keep the first 90% of whole stories, in file
order, for training and the rest as held-out validation (24,867 / 2,763).
Both splits are encoded with Karpathy's tok512 byte-pair encoding; the
end-of-sequence token sits between stories, and the stream starts with it,
so each story's first token is conditioned on end-of-sequence.
"""

from __future__ import annotations

import os

import numpy as np

WEEK8_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(WEEK8_DIR, "data", "TinyStoriesV2-GPT4-valid.txt")
TOKENIZER_PATH = os.path.join(WEEK8_DIR, "inputs", "tinyllamas", "tok512.model")

SEPARATOR = "<|endoftext|>"
V = 512


def load_stories_from_text(text: str) -> list[str]:
    """Split at <|endoftext|>, strip whitespace, drop empty stories."""
    return [s.strip() for s in text.split(SEPARATOR) if s.strip()]


def load_stories(path: str = CORPUS_PATH) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return load_stories_from_text(f.read())


def split_stories(stories: list[str], frac: float = 0.9):
    """First frac of whole stories, in file order, train; the rest validate."""
    n_train = int(round(frac * len(stories)))
    return stories[:n_train], stories[n_train:]


def load_tokenizer(path: str = TOKENIZER_PATH):
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(model_file=path)
    assert sp.piece_size() == V
    return sp


def eos_id(sp) -> int:
    return sp.eos_id()


def encode_stream(sp, stories: list[str], eos_id: int) -> np.ndarray:
    """[EOS] + s1 + [EOS] + s2 + [EOS] + ...  (no trailing EOS after the last)."""
    parts = [np.asarray([eos_id], dtype=np.int32)]
    for i, story in enumerate(stories):
        parts.append(np.asarray(sp.encode(story), dtype=np.int32))
        if i + 1 < len(stories):
            parts.append(np.asarray([eos_id], dtype=np.int32))
    return np.concatenate(parts)


def cache_streams():
    """Encode both splits once and cache the token streams to artifacts/."""
    out_dir = os.path.join(WEEK8_DIR, "artifacts", "part2")
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_tokens.npy")
    val_path = os.path.join(out_dir, "val_tokens.npy")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return np.load(train_path), np.load(val_path)
    sp = load_tokenizer()
    eos = eos_id(sp)
    train_stories, val_stories = split_stories(load_stories())
    train = encode_stream(sp, train_stories, eos)
    val = encode_stream(sp, val_stories, eos)
    np.save(train_path, train)
    np.save(val_path, val)
    return train, val


if __name__ == "__main__":
    sp = load_tokenizer()
    eos = eos_id(sp)
    train_stories, val_stories = split_stories(load_stories())
    print(f"stories: {len(train_stories)} train / {len(val_stories)} validation")
    train, val = cache_streams()
    print(f"tokens: {len(train):,} train / {len(val):,} validation")

    # Decode check: every validation story should round-trip except ~3 in
    # which the tokenizer merged a double space into one.
    n_ok = 0
    bad = []
    pos = 1  # stream starts with EOS
    for i, story in enumerate(val_stories):
        n = len(sp.encode(story))
        toks = val[pos:pos + n]
        decoded = sp.decode(toks.tolist())
        if decoded == story:
            n_ok += 1
        else:
            bad.append(i)
        pos += n + 1  # story tokens + EOS separator
    print(f"decode check: {n_ok}/{len(val_stories)} validation stories "
          f"round-trip exactly; differing indices {bad}")
    chars = sum(len(s) for s in val_stories)
    print(f"validation characters: {chars:,}; chars per token: "
          f"{chars / len(val):.3f} (sheet ~2.05)")
