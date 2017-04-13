import pytest
import numpy as np
import Antipasti.backend as A


def test_shuffle_tensor():
    # Make numerics
    numerical_tensor = np.random.uniform(size=(2, 5, 5, 2))

    # Case 1: Known ndim, axis = -1
    possible_numerical_outputs = [numerical_tensor, numerical_tensor[..., ::-1]]
    # Make tensor and op
    tensor = A.placeholder(shape=[None] * 4)
    shuffled_tensor = A.shuffle_tensor(tensor, axis=-1)
    # Evaluate
    output = A.run(shuffled_tensor, {tensor: numerical_tensor})
    # Check if output checks out
    assert any([np.allclose(numerical_output, output)
                for numerical_output in possible_numerical_outputs])

    # Case 2: unknown ndim, axis = 0
    possible_numerical_outputs = [numerical_tensor, numerical_tensor[::-1, ...]]
    # Make tensor and op
    tensor = A.placeholder()
    shuffled_tensor = A.shuffle_tensor(tensor)
    # Evaluate
    output = A.run(shuffled_tensor, {tensor: numerical_tensor})
    # Check if output checks out
    assert any([np.allclose(numerical_output, output)
                for numerical_output in possible_numerical_outputs])

    # Case 3: Unknown ndim, axis = -1
    with pytest.raises(AssertionError):
        # Make tensor and op
        tensor = A.placeholder()
        shuffled_tensor = A.shuffle_tensor(tensor, axis=-1)

    # Case 4: Known dim, axis >= ndim
    with pytest.raises(AssertionError):
        # Make tensor and op
        tensor = A.placeholder(shape=[None] * 4)
        shuffled_tensor = A.shuffle_tensor(tensor, axis=4)


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


if __name__ == '__main__':
    pytest.main([__file__])