import numpy as np
import pytest

from stories import encode_stream, load_stories_from_text, split_stories


TEXT = (
    "<|endoftext|>\n  A story.  \n<|endoftext|>\n\n<|endoftext|>Second story here.<|endoftext|>\n"
)


def test_split_strips_and_drops_empty():
    stories = load_stories_from_text(TEXT)
    assert stories == ["A story.", "Second story here."]


def test_split_first_90_percent_by_file_order():
    stories = [f"s{i}" for i in range(10)]
    train, val = split_stories(stories, frac=0.9)
    assert train == [f"s{i}" for i in range(9)]
    assert val == ["s9"]


def test_encode_stream_starts_with_eos_and_separates_stories():
    class StubSP:
        def encode(self, text):
            return {"aa": [7], "bb": [8, 9]}[text]

    tokens = encode_stream(StubSP(), ["aa", "bb"], eos_id=2)
    assert tokens.tolist() == [2, 7, 2, 8, 9]  # EOS between stories, none after the last


def test_encode_stream_dtype_and_1d():
    class StubSP:
        def encode(self, text):
            return [1]

    tokens = encode_stream(StubSP(), ["x", "y", "z"], eos_id=0)
    assert tokens.dtype == np.int32
    assert tokens.ndim == 1


@pytest.mark.skipif(
    not __import__("os").path.exists(
        __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.dirname(
                __import__("os").path.abspath(__file__))),
            "data", "TinyStoriesV2-GPT4-valid.txt")),
    reason="corpus not downloaded",
)
def test_real_corpus_split_counts():
    import stories

    train, val = split_stories(stories.load_stories(stories.CORPUS_PATH))
    assert len(train) == 24867
    assert len(val) == 2763
