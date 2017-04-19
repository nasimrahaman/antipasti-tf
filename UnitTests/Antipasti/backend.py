import pytest
import threading

import numpy as np
import Antipasti.backend as A
import tensorflow as tf


def test_with_master_graph():
    # Get master graph
    master_graph = tf.get_default_graph()

    # Make function in a different thread that checks if the graphs are same
    @A.with_master_graph
    def check():
        graph_in_thread = tf.get_default_graph()
        assert graph_in_thread is master_graph

    # Make adn Start thread
    thread = threading.Thread(target=check)
    thread.start()
    thread.join()


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

    # Check whether the gradients can be computed
    tensor = A.placeholder()
    shuffled_tensor = A.shuffle_tensor(tensor, differentiable=True)
    grad_tensor = A.gradients(objective=A.reduce_(shuffled_tensor, 'mean'), with_respect_to=tensor)
    numerical_grad_tensor = A.run(grad_tensor, {tensor: numerical_tensor})
    assert numerical_grad_tensor.values.shape == numerical_tensor.shape

    # Check whether the lookup error is raised otherwise
    with pytest.raises(LookupError):
        tensor = A.placeholder()
        shuffled_tensor = A.shuffle_tensor(tensor, differentiable=False)
        grad_tensor = A.gradients(objective=A.reduce_(shuffled_tensor, 'mean'),
                                  with_respect_to=tensor)

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