import numpy as np
import Antipasti.backend as A


def test_sorensen_dice_distance():
    _y = np.zeros(shape=(5,))
    _yt = np.zeros(shape=(5,))
    _y[0:2] = 1.
    _yt[0:2] = 1.
    y = A.placeholder()
    yt = A.placeholder()
    d = A.sorensen_dice_distance(y, yt, with_logits=False)
    _d = A.run(d, {y: _y, yt: _yt})
    assert _d == 0.


def test_tversky_distance():
    _y = np.zeros(shape=(5,))
    _yt = np.zeros(shape=(5,))
    _y[0:2] = 1.
    _yt[0:2] = 1.
    y = A.placeholder()
    yt = A.placeholder()
    d = A.tversky_distance(y, yt, with_logits=False)
    _d = A.run(d, {y: _y, yt: _yt})
    assert _d == 0.
