__author__ = "Nasim Rahaman"

from collections import OrderedDict


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


class ParameterCollection(OrderedDict):
    # TODO: Can of worms for another day.
    pass
