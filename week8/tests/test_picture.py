import numpy as np

from picture import SUAN_BITMAP, recall, target_picture
from stores import softmax_R_prediction


def test_bitmap_is_32x32_with_249_ink_pixels():
    assert len(SUAN_BITMAP) == 32
    for row in SUAN_BITMAP:
        assert 0 <= row < 2 ** 32
    pic = target_picture()
    assert pic.shape == (1024,)
    assert set(np.unique(pic)) <= {-1.0, 1.0}
    assert int((pic == 1).sum()) == 249          # extracted from sheet fig. 13


def test_softmax_recall_of_a_lone_memory_is_exact():
    rng = np.random.default_rng(0)
    recalled, R = recall(M=1, d=64, store=("softmax", 8.0), rng=rng)
    # no other memories: the answer is the picture itself (own weight removed)
    assert R < 1e-10
    assert np.allclose(recalled, target_picture(), atol=1e-9)


def test_recall_R_matches_predictions_at_M_64():
    # outer: sqrt(M/d) = 1; relu2: sqrt(3M/(2 d^p)) = sqrt(3*64/(2*64^2)) = 0.153
    rng = np.random.default_rng(1)
    _, r_out = recall(M=64, d=64, store="outer", rng=rng)
    rng = np.random.default_rng(2)
    _, r_relu2 = recall(M=64, d=64, store=("relu", 2), rng=rng)
    assert abs(r_out - np.sqrt(64 / 64)) < 0.35
    assert abs(r_relu2 - 0.153) < 0.08


def test_softmax_prediction_formula_matches_sheet_value():
    # sheet: at M=4096, d=64, beta=8 the prediction is 0.058
    assert abs(softmax_R_prediction(4096, 64, 8.0) - 0.058) < 0.001
