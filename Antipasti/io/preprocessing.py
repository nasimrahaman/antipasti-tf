__author__ = "Nasim Rahaman"

from itertools import product
import numpy as np


def as_function_over_axes(axes):
    """
    Returns a decorator that applies a provided input function only over the given `axes` of the
    batch tensor.

    Example:
        ```
        @as_function_over_axes((1, 2))
        def print_shape(input_):
            print(input_.shape)

        input_array = numpy.zeros(shape=(2, 100, 150, 3))
        print_shape(input_array)

        # Prints (100, 150) two (= shape[0]) * three (=shape[3]) = six times:
        #   (100, 150)
        #   (100, 150)
        #   (100, 150)
        #   (100, 150)
        #   (100, 150)
        #   (100, 150)

        ```
    :type axes: list or tuple
    :param axes: Axes to distribute the function over.

    :return: Decorator to apply a function over the given axes.
    """

    def decorator(function):
        def _new_function(batch_in):
            # Validate
            assert isinstance(batch_in, np.ndarray), \
                "Input batch must be a numpy ndarray, " \
                "got {} instead.".format(batch_in.__class__.__name__)
            # Take measurements
            batch_shape = batch_in.shape
            batch_ndim = len(batch_shape)
            # Generate a iterable of slices to loop over.
            slices_to_loop_over = product(*[range(batch_shape[dim_num])
                                            if dim_num not in axes else
                                            [slice(None)]
                                            for dim_num in range(batch_ndim)])
            # Preallocate output
            batch_out = np.zeros_like(batch_in)
            # Start main loop
            for _slice in slices_to_loop_over:
                batch_out[_slice] = function(batch_in[_slice])
            # Done.
            return batch_out
        return _new_function
    return decorator

