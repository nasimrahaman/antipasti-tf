__author__ = "Nasim Rahaman"

from ..legacy import pyutils as py
from . import Layer
from .. import backend as A
from .. import utils

try:
    import keras
except ImportError as err:
    keras = None
    raise ImportError("Keras could not be imported. The original error "
                      "message follows: {}".format(err.message))


class KerasLayer(Layer):
    """Layer to wrap around a Keras layer or a model."""
    def __init__(self, input, output, lock_shapes=False, **layer_kwargs):
        """
        :type input: list or any
        :param input: Input variable or a list of input variables, each returned by keras.layers.Input.

        :type output: list or any
        :param output: Output variable or a list of output variables, each returned by a Keras model.

        :type lock_shapes: bool
        :param lock_shapes: Whether to not allow deviation from the set input_shape while building the model.

        :type layer_kwargs: dict
        :param layer_kwargs: Keyword arguments for the superclass.
        """
        super(KerasLayer, self).__init__(**layer_kwargs)
        self.keras_input = input
        self.keras_output = output
        self.lock_shapes = lock_shapes
        self.keras_model = keras.models.Model(input=self.keras_input, output=self.keras_output, name=self.name)
        self.input_shape = utils.get_input_shape(
            known_input_shape=to_antipasti_shape(get_keras_shape(self.keras_input)))

    @property
    def parameters(self):
        return self.keras_model.weights

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        # If shapes are locked, check
        if self.lock_shapes:
            keras_input_shapes = py.list2listoflists(self.input_shape)
            given_input_shapes = py.list2listoflists(input_shape)
            for input_num, (keras_input_shape, given_input_shape) in enumerate(zip(keras_input_shapes,
                                                                                   given_input_shapes)):
                if not utils.compare_shapes(keras_input_shape, given_input_shape):
                    raise ValueError(self._stamp_string("Given shape of the {}-th input (= {}) is not "
                                                        "consistent with what Keras expects (= {}). "
                                                        "Set lock_shapes field to False if this is intentional.".
                                                        format(input_num, given_input_shape, keras_input_shape)))
        # Infer output shape with Keras. Keras expects shapes as [(...), (...), (...), ...] whereas Antipasti
        # works with [[...], [...], [...], ...]. Feeding in a list [...] confuses keras (see to_list function in
        # keras.engine.topology) and makes it think that there are multiple inputs.
        output_shape = self.keras_model.get_output_shape_for(to_keras_shape(input_shape))
        # Remember, keras works with lists of tuples, we need lists of lists
        output_shape = to_antipasti_shape(output_shape)
        return output_shape

    @utils.forward_pass
    def feedforward(self, input=None):
        output = self.keras_model(input)
        return output


class AntipastiLayer(keras.engine.topology.Layer):
    def __init__(self, model, **kwargs):
        self.antipasti_model = model
        super(AntipastiLayer, self).__init__(name=self.antipasti_model.name, **kwargs)

    def build(self, input_shape):
        self.antipasti_model.input_shape = input_shape
        # TODO Cleaner implementation with parameter tag check (to check if parameters are trainable)
        self.trainable_weights = self.antipasti_model.parameters
        self.built = True

    def call(self, x, mask=None):
        return self.antipasti_model.feedforward(input=x)

    def get_output_shape_for(self, input_shape):
        antipasti_shape = self.antipasti_model.infer_output_shape(input_shape=input_shape)


# ---- Helper functions
def get_keras_shape(variables):
    """Try to get variable shape(s) from Keras, and fall back to tensorflow if that fails."""
    shapes = []
    for variable in py.obj2list(variables):
        if hasattr(variable, '_keras_shape'):
            # Try to get keras shape
            shapes.append(variable._keras_shape)
        else:
            # Keras shape is not available. Get tensorflow shape
            var_shape = A.shape(variable)
            if var_shape is None:
                raise ValueError("Shape could not be infered for tensorflow"
                                 "variable/placeholder named '{}'.".format(variable.name))
            shapes.append(var_shape)
    return py.delistlistoflists(shapes)


def to_keras_model(model):
    """Convert an Antipasti Model to a Keras model (with a few strings attached)."""
    # Get model input(s)
    model_inputs = []
    input_shapes = py.list2listoflists(model.input_shape)
    for _input_num in range(model.num_inputs):
        model_inputs.append(keras.layers.Input(shape=input_shapes[_input_num][1:],
                                               batch_shape=input_shapes[_input_num][0]))
    model_input = py.delist(model_inputs)

    # Build keras layer with the model and get its output
    layer = AntipastiLayer(model=model)
    model_output = layer(model_inputs)
    # Build Keras model with input and output
    keras_model = keras.models.Model(input=model_input, output=model_output)

    # Done.
    return keras_model


def to_keras_shape(shape):
    if py.islistoflists(shape):
        return py.listoflists2listoftuples(shape)
    elif shape is not None:
        return tuple(shape)
    else:
        raise ValueError("Can't convert `None` to Keras shape.")


def to_antipasti_shape(shape):
    if py.islistoflists(shape):
        return py.listoftuples2listoflists(shape)
    elif shape is not None:
        return list(shape)
    else:
        raise ValueError("Can't convert `None` to Antipasti shape.")
