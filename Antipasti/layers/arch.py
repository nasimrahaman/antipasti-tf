__author__ = "Nasim Rahaman"

from ..utilities import utils
from .core import Layer
from .. import backend as A
from ..legacy import pykit as py


class ReplicateLayer(Layer):
    """Layer that replicates its inputs a given `num_replicate` number of times."""
    def __init__(self, num_replicate, dimensions=None, input_shape=None, **layer_kwargs):
        # Init superclass
        super(ReplicateLayer, self).__init__(**layer_kwargs)
        # Attach meta
        self.num_replicate = num_replicate
        # Get input_shape signature and run shape inference
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=1, known_input_shape=input_shape,
                                                 _string_stamper=self._stamp_string)

    @utils.forward_pass
    def feedforward(self, input=None):
        # Replicate and return
        return [input] * self.num_replicate

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        return [input_shape] * self.num_replicate


class ConcatenateLayer(Layer):
    """Layer that concatenates its inputs along a given axis."""
    def __init__(self, num_inputs=None, axis=-1, dimensions=None, input_shape=None, **layer_kwargs):
        # Init superclass
        super(ConcatenateLayer, self).__init__(**layer_kwargs)

        # Attach meta
        assert axis in [0, 1, 2, 3, 4, -1], self._stamp_string("Supported axis are [0, 1, 2, 3, 4, -1], got {}.".
                                                               format(axis))
        self.axis = axis

        # Get input shape
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=num_inputs,
                                                 known_input_shape=input_shape, _string_stamper=self._stamp_string)

    @utils.forward_pass
    def feedforward(self, input=None):
        return A.concatenate(input, axis=self.axis)

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        # Check input shape for consistency
        input_shape_ok = py.islistoflists(input_shape)
        if self.axis == -1:
            input_shape_ok = input_shape_ok and \
                             all([utils.compare_shapes(ishp[:self.axis], input_shape[0][:self.axis])
                                  for ishp in input_shape])
        else:
            input_shape_ok = input_shape_ok and \
                             all([utils.compare_shapes(ishp[:self.axis] + ishp[(self.axis + 1):],
                                                       input_shape[0][:self.axis] + input_shape[0][(self.axis + 1):])
                                  for ishp in input_shape])

        if not input_shape_ok:
            raise ValueError(self._stamp_string("Given input shape {} is not consistent "
                                                "for concatenation along axis = {}. Check "
                                                "if the shapes of the individual inputs "
                                                "are equal in all but the {}-th dimension.".format(input_shape,
                                                                                                   self.axis,
                                                                                                   self.axis)))
        # Compute output shape
        output_shape = input_shape[0][0:self.axis] + \
                       [sum([ishp[self.axis] for ishp in input_shape])
                        if None not in [ishp[self.axis] for ishp in input_shape]
                        else None] + \
                       (input_shape[0][(self.axis + 1):] if self.axis != -1 else [])
        return output_shape


class AddLayer(Layer):
    """Layer that adds its inputs."""
    def __init__(self, num_inputs=None, dimensions=None, input_shape=None, **layer_kwargs):
        # Initialize superclass
        super(AddLayer, self).__init__(**layer_kwargs)
        # Set input shape
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=num_inputs,
                                                 known_input_shape=input_shape, _string_stamper=self._stamp_string)
        # Make sure the number of inputs is larger than 1
        if not self.num_inputs > 1:
            raise ValueError(self._stamp_string("AddLayer expects more than one inputs (got 1). "
                                                "If no `input_shape` is given or if `dimensions` "
                                                "is a scalar, please provide `num_inputs`."))

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        # Validate input_shape
        assert py.islistoflists(input_shape), \
            self._stamp_string("AddLayer expects more than 1 inputs, which implies "
                               "that the `input_shape` should be a list of lists.")
        assert len(input_shape) == self.num_inputs, \
            self._stamp_string("AddLayer expects {} inputs, which is not consistent with "
                               "the provided shape signature implying {} inputs.".
                               format(len(input_shape), self.num_inputs))
        assert all([utils.compare_shapes(_input_shape, input_shape[0], soft=True) for _input_shape in input_shape]), \
            self._stamp_string("All inputs to the AddLayer must have the same shape "
                               "(except for None's). The following `input_shape` is "
                               "therefore not compatible: {}.".format(input_shape))
        # Get output shape and return
        output_shape = input_shape[0]
        return output_shape

    @utils.forward_pass
    def feedforward(self, input=None):
        return A.add_n(tensors=input)


class IdentityLayer(Layer):
    """Layer that does nothing to its input."""
    def __init__(self, dimensions=None, input_shape=None, **layer_kwargs):
        super(IdentityLayer, self).__init__(**layer_kwargs)
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=1, known_input_shape=input_shape)

    @utils.forward_pass
    def feedforward(self, input=None):
        return input

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        return input_shape


class FunctionLayer(Layer):
    """Layer that applies a given function to its input."""
    def __init__(self, function, shape_inference_function=None, parameters=None,
                 num_inputs=None, dimensions=None, input_shape=None, **layer_kwargs):
        super(FunctionLayer, self).__init__(**layer_kwargs)

        # Make sure the functions are callable
        if not callable(function):
            raise ValueError(self._stamp_string("The argument `function` must be a callable object."))

        if shape_inference_function is not None and not callable(shape_inference_function):
            raise ValueError(self._stamp_string("The argument `shape_inference_function` must be "
                                                "None or a callable object."))

        # Assign attributes
        self.function = function
        self.shape_inference_function = shape_inference_function \
            if shape_inference_function is not None else lambda _input_shape: _input_shape

        self.input_shape = utils.get_input_shape(dimensions=dimensions, known_input_shape=input_shape,
                                                 num_inputs=num_inputs)
        # Register parameters (assumed present in the function closure, i.e. not fed in to the function at
        # feedforward-time)
        for parameter in parameters:
            self.register_parameter(parameter)

    @utils.forward_pass
    def feedforward(self, input=None):
        return self.function(input)

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        return self.shape_inference_function(input_shape)


class SliceDistributedLayer(Layer):
    """
    Layer for distributing 2D layers along 'DH', 'HW' or 'DW' slices of a 'BDHWC' tensor, where
        D = Depth,
        H = Height
        W = Width
        B = Batch
        C = Channel
    """

    _MAP_APPLY_ON_TO_IMAGE_AXIS = {'HW': 0, 'DW': 1, 'DH': 2}

    def __init__(self, layer, apply_on='HW', num_slices=None, **layer_kwargs):
        # Initialize superclass
        super(SliceDistributedLayer, self).__init__(**layer_kwargs)
        # Private
        self._apply_on = None

        # Assignments
        self.num_slices = num_slices
        self.apply_on = apply_on
        self.child_layer = layer

        # Shape inference
        self.input_shape = self.get_input_shape_from_child()

    def get_input_shape_from_child(self, child_layer=None):
        if child_layer is None:
            assert hasattr(self, 'child_layer'), \
                self._stamp_string("The child_layer is yet to be defined.")
            child_layer = self.child_layer

        # Make sure the child is 2D
        assert child_layer.dimensions == 2, \
            self._stamp_string("The `child_layer` must be 2 dimensional, but it's {}D.".
                               format(self.child_layer.dimensions))

        input_shape = child_layer.input_shape[:]
        input_shape.insert(self.tensor_axis, self.num_slices)
        return input_shape

    @property
    def apply_on(self):
        return self._apply_on

    @apply_on.setter
    def apply_on(self, value):
        if value not in self._MAP_APPLY_ON_TO_IMAGE_AXIS:
            raise ValueError(self._stamp_string("`apply_on` must be in {}. Got {} instead.".
                                                format(self._MAP_APPLY_ON_TO_IMAGE_AXIS.keys(), value)))
        self._apply_on = value

    @property
    def image_axis(self):
        return self._MAP_APPLY_ON_TO_IMAGE_AXIS.get(self.apply_on)

    @image_axis.setter
    def image_axis(self, value):
        _INV_MAP_APPLY_ON_TO_IMAGE_AXIS = {val: key for key, val in self._MAP_APPLY_ON_TO_IMAGE_AXIS.items()}
        # Validate value
        if value not in _INV_MAP_APPLY_ON_TO_IMAGE_AXIS.keys():
            raise ValueError(self._stamp_string("`image_axis` must be in {}. Got {} instead.".
                                                format(_INV_MAP_APPLY_ON_TO_IMAGE_AXIS.keys(), value)))
        # Set apply_on
        self.apply_on = _INV_MAP_APPLY_ON_TO_IMAGE_AXIS[value]

    @property
    def tensor_axis(self):
        # Infer from image_axis (add one for the batch axis)
        return self.image_axis + 1

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        # Check if num_slices can be inferred anew
        if input_shape[self.tensor_axis] is not None:
            self.num_slices = input_shape[self.tensor_axis]

        # Infer output_shape from child
        child_input_shape = input_shape[:]
        child_input_shape.pop(self.tensor_axis)
        child_output_shape = self.child_layer.infer_output_shape(input_shape=child_input_shape)[:]
        # Process infered output shape
        output_shape = child_output_shape.insert(self.tensor_axis, self.num_slices)
        return output_shape

    @utils.layer_initialization
    def initialize_layer(self, input_shape=None):
        self.child_layer.initialize_layer(input_shape=input_shape)

    @utils.forward_pass
    def feedforward(self, input=None):
        # The more scenic route would be to use the inferred output shape, but we avoid doing that for robustness.
        # Get axes over which the child layer is NOT parallelized
        child_layer_axes = [1, 2, 3]
        child_layer_axes.remove(self.tensor_axis)
        # Transpose input
        input_transposed = A.transpose(input,
                                       perm=[0, self.tensor_axis, child_layer_axes[0], child_layer_axes[1], 4])
        # Get symbolic shape of the transposed input
        input_transposed_shape = A.shape(input_transposed, symbolic=True)
        batch_size = input_transposed_shape[0]
        num_slices = input_transposed_shape[1]
        # Tensorflow doesn't support '+' operator overloading for list concatenation
        input_transposed_reshaped = A.reshape(input_transposed,
                                              shape=[-1,
                                                     input_transposed_shape[2],
                                                     input_transposed_shape[3],
                                                     input_transposed_shape[4]])
        # Apply child layer and get the symbolic shape of the resulting tensor
        output_transposed_reshaped = self.child_layer.feedforward(input=input_transposed_reshaped)
        output_transposed_reshaped_shape = A.shape(output_transposed_reshaped, symbolic=True)
        # Undo reshape
        output_transposed = A.reshape(output_transposed_reshaped, shape=[batch_size, num_slices,
                                                                         output_transposed_reshaped_shape[1],
                                                                         output_transposed_reshaped_shape[2],
                                                                         output_transposed_reshaped_shape[3]])
        # Undo transpose
        output = A.transpose(output_transposed,
                             perm=[0, self.tensor_axis, child_layer_axes[0], child_layer_axes[1], 4])
        # Done
        return output
