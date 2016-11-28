__author__ = "Nasim Rahaman"

from .core import Layer
from .. import utils


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

    def feedforward(self, input=None):
        if input is None:
            input = self.x
        else:
            self.x = input
        # TODO: Do we need a context manager here?
        # Replicate and return
        self.y = [input] * self.num_replicate
        return self.y

    def infer_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        return [input_shape] * self.num_replicate
