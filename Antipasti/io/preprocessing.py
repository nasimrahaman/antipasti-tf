__author__ = "Nasim Rahaman"

from itertools import product
import numpy as np

from ..legacy import pykit as py


def as_function_over_axes(axes):
    """
    Returns a decorator that applies a provided input function only over the given `axes` of the
    batch tensor. The function to be decorated must take one or more arguments and return exactly
    as many outputs as arguments. Arguments and outputs must all be numpy ndarrays of the same
    shape.

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
        def _new_function(batches_in):
            batches_in = py.obj2list(batches_in, ndarray2list=False)
            # Validate
            for batch_in in batches_in:
                assert isinstance(batch_in, np.ndarray), \
                    "Input batch must be a numpy ndarray, " \
                    "got {} instead.".format(batch_in.__class__.__name__)
                assert batch_in.shape == batches_in[0].shape, \
                    "All input batches must have the same shape. " \
                    "Shape {} is incompatible with {}.".format(batch_in.shape, batches_in[0].shape)

            # Take measurements
            batch_shape = batches_in[0].shape
            batch_ndim = len(batch_shape)

            # Generate a iterable of slices to loop over.
            slices_to_loop_over = product(*[range(batch_shape[dim_num])
                                            if dim_num not in axes else
                                            [slice(None)]
                                            for dim_num in range(batch_ndim)])

            # Preallocate output(s)
            batches_out = [np.zeros_like(batch_in) for batch_in in batches_in]

            # Start main loop
            for _slice in slices_to_loop_over:
                # Slice all batches
                sliced_batches_in = [batch_in[_slice] for batch_in in batches_in]
                # Get output and convert it to a list in case it isn't one already
                all_outputs = py.obj2list(function(py.delist(sliced_batches_in)),
                                          ndarray2list=False)
                for given_batch_out, given_output in zip(batches_out, all_outputs):
                    given_batch_out[_slice] = given_output
            # Delist and return
            return py.delist(batches_out)
        return _new_function
    return decorator

