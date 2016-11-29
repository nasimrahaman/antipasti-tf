__author__ = "Nasim Rahaman"

from .core import Layer
from .. import utils
from .. import backend as A


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
        # Get input and output tensors
        xy_placeholders = utils.get_layer_xy_placeholders(input_shape=self.input_shape, output_shape=self.output_shape,
                                                          layer_id=self.name, device=self.device,
                                                          variable_scope=self.variable_scope,
                                                          context_managers=self.context_managers)
        # Write layer x and y
        self.x = xy_placeholders['x']
        self.y = xy_placeholders['y']

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
        self.axis = axis

        # Get input shape
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=num_inputs,
                                                 known_input_shape=input_shape)
        # Get input and output tensors
        xy_placeholders = utils.get_layer_xy_placeholders(input_shape=self.input_shape, output_shape=self.output_shape,
                                                          layer_id=self.name, device=self.device,
                                                          variable_scope=self.variable_scope,
                                                          context_managers=self.context_managers)
        # Write layer x and y
        self.x = xy_placeholders['x']
        self.y = xy_placeholders['y']

    @utils.forward_pass
    def feedforward(self, input=None):
        return A.concatenate(input, axis=self.axis)

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        # TODO
        pass
