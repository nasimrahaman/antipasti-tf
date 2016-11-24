__author__ = "Nasim Rahaman"

from collections import OrderedDict
from contextlib2 import ExitStack

from . import pyutils as py


def forward_pass(forward_function):
    """
    Decorator for the feedforward method of `Layer`. The `feedforward` method must be able to handle input as a
    keyword argument; this decorator ensures that any given `forward_function` is always fed its input - in other
    words, `forward_function` can be allowed to handle its input as an argument (and not necessarily a keyword
    argument).
    """
    def _feedforward(cls, input=None):
        # Define behaviour for when input is None:
        if input is None:
            input = cls.x
        else:
            cls.x = input
        # Evaluate output
        output = forward_function(cls, input=input)
        # Assign output to y
        cls.y = output
        # Return output
        return output
    # Return decorated function
    return _feedforward


def shape_inference(shape_inference_function):
    """
    Decorator for the `infer_output_shape` method of `Layer`. The motivation for this decorator is the same as that
    of the `forward_pass` decorator.
    """

    def _infer_output_shape(cls, input_shape=None):
        if input_shape is None:
            input_shape = cls.input_shape
        return shape_inference_function(cls, input_shape=input_shape)

    return _infer_output_shape


def call_in_managers(context_managers=None):
    """
    Decorator factory that makes a decorator to call the decorated function within nested `context_managers`.

    :type context_managers: list
    :param context_managers: List of context managers to nest over. The first manager in list is entered first.
    """
    def _decorator(function):
        def decorated_function(*args, **kwargs):
            with ExitStack as stack:
                # Enter managers
                for manager in context_managers:
                    stack.enter_context(manager)
                # Evaluate function
                output = function(*args, **kwargs)
            return output
        return decorated_function
    return _decorator


def get_input_shape(dimensions=None, known_input_shape=None, num_features_in=None, batch_size=None,
                    _string_stamper=None):
    """Deduce input_shape (in NHWC (byxc) or NDHWC (bzyxc) format) from what's known."""

    if _string_stamper is None:
        _string_stamper = lambda s: s

    _dimension_to_tensor_dimension = {2: 4, 3: 5}

    # If an input shape is given:
    if known_input_shape is None:
        assert dimensions is not None, _string_stamper("Need `dimensions` argument to infer input_shape.")
        known_input_shape = [None for _ in range(_dimension_to_tensor_dimension[dimensions])]

    assert len(known_input_shape) == 4 or len(known_input_shape) == 5, \
        _string_stamper("input_shape must be 4 or 5 elements long, not {}.".format(len(known_input_shape)))

    if num_features_in is not None:
        assert known_input_shape[-1] is None or known_input_shape[-1] == num_features_in, \
            _string_stamper("Given number of input features ({}) is not consistent with the given input_shape "
                            "(expecting {} input features).".format(num_features_in, known_input_shape[-1]))
        known_input_shape[-1] = num_features_in

    if batch_size is not None:
        assert known_input_shape[0] is None or known_input_shape[0] == batch_size, \
            _string_stamper("Given number of input features ({}) is not consistent with the given input_shape "
                            "(expecting {} input features).".format(num_features_in, known_input_shape[-1]))
        known_input_shape[0] = batch_size

    return known_input_shape


class ParameterCollection(OrderedDict):
    """Class to collect parameters of a layer."""
    # TODO: Can of worms for another day.
    pass
