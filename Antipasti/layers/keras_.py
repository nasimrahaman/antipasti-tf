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
        return antipasti_shape


# ---- Keras layers to-go

def conv(maps_in, maps_out, kernel_size, stride=None, dilation=None, border_mode='same', input_shape=None, name=None,
         **keras_kwargs):
    """Make a convolutional layer with Keras."""

    # Get the number of dimensions from kernel_size
    dimensions = len(kernel_size)
    # Parse input shape
    input_shape = utils.get_input_shape(dimensions=dimensions, known_input_shape=input_shape, num_inputs=1,
                                        num_features_in=maps_in)

    # Set default stride if required
    subsample = tuple(stride) if stride is not None else (1,) * dimensions
    assert len(subsample) == dimensions, "Stride must be a tuple of the same length as kernel_size; " \
                                         "expected {}, found {}.".format(dimensions, len(subsample))

    # Set default dilation if required
    rate = tuple(dilation) if dilation is not None else (1,) * dimensions

    # A few consistency checks
    is_strided = subsample != (1, 1) or subsample != (1, 1, 1)
    is_dilated = dilation != (1, 1) or dilation != (1, 1, 1)
    is_3D = dimensions == 3
    # Dilated convolutions support in 2D only
    assert not (is_3D and is_dilated), "No support for 3D dilated convolutions yet."
    # It's either strided convolution or 2D dilated convolution, but not both
    assert not (is_strided and is_dilated), "It's either strided or dilated (atrous) convolution, but not both."

    # Make input to the Keras layer
    keras_input = keras.layers.Input(shape=input_shape[1:], batch_shape=input_shape[0])

    # Make Keras convolutional layer
    if not is_dilated:
        # 2D or 3D?
        if not is_3D:
            # 2D
            keras_convlayer = keras.layers.Convolution2D(nb_filter=maps_out,
                                                         nb_row=kernel_size[0], nb_col=kernel_size[1],
                                                         border_mode=border_mode, subsample=subsample,
                                                         **keras_kwargs)
        else:
            # 3D
            keras_convlayer = keras.layers.Convolution3D(nb_filter=maps_out,
                                                         kernel_dim1=kernel_size[0], kernel_dim2=kernel_size[1],
                                                         kernel_dim3=kernel_size[2],
                                                         border_mode=border_mode, subsample=subsample,
                                                         **keras_kwargs)
    else:
        # 2D dilated convolution
        keras_convlayer = keras.layers.AtrousConvolution2D(nb_filter=maps_out,
                                                           nb_row=kernel_size[0], nb_col=kernel_size[1],
                                                           border_mode=border_mode, atrous_rate=rate,
                                                           **keras_kwargs)

    # Build Keras graph
    keras_output = keras_convlayer(keras_input)
    # Build KerasLayer from keras_graph
    layer = KerasLayer(input=keras_input, output=keras_output, name=name)

    # Attach meta information (to mimic that of the future convolutional layer)
    layer.maps_in = maps_in
    layer.maps_out = maps_out
    layer.kernel_size = kernel_size
    layer.stride = subsample
    layer.dilation = rate
    layer.border_mode = border_mode
    layer.keras_kwargs = keras_kwargs

    # Done.
    return layer


def pool(window, stride=None, pool_mode='max', global_=False, border_mode='same', input_shape=None, name=None,
         **keras_kwargs):
    """Make a pooling layer with Keras."""

    assert pool_mode in {'max', 'mean'}, "Invalid `pool_mode` '{}'; allowed are 'max' and 'mean'.".format(pool_mode)

    # Infer the number of dimensions from the given window
    dimensions = len(window)
    assert dimensions in {2, 3}, "Dimension {} is not supported. Supported dimensions are 2 and 3.".format(dimensions)

    # Parse input shape
    input_shape = utils.get_input_shape(dimensions=dimensions, known_input_shape=input_shape, num_inputs=1)

    # Set default stride if required
    stride = tuple(stride) if stride is not None else tuple(window)
    assert len(stride) == dimensions, "Stride must be a tuple of the same length as kernel_size; " \
                                      "expected {}, found {}.".format(dimensions, len(stride))

    # Get input to the keras layer
    keras_input = keras.layers.Input(shape=input_shape[1:], batch_shape=input_shape[0])

    # Make Keras pooling layer
    if not global_:
        # Vanilla pooling:
        # Construct pooling class name
        pool_class_name = "{}Pooling{}D".format({'max': 'Max', 'mean': 'Average'}.get(pool_mode),
                                                dimensions)
        keras_poollayer = getattr(keras.layers, pool_class_name)(pool_size=window, strides=stride,
                                                                 border_mode=border_mode, **keras_kwargs)
        # The shapes are correct, nothing to do
        keras_postprocessing_layer = keras.layers.Activation(activation='linear')
    else:
        # Global pooling
        # Construct pooling class name
        pool_class_name = "Global{}Pooling{}D".format({'max': 'Max', 'mean': 'Average'}.get(pool_mode),
                                                      dimensions)
        keras_poollayer = getattr(keras.layers, pool_class_name)(**keras_kwargs)
        # Global pooling in Keras gets rid of the last two spatial dimensions. We don't want this, so we add in an
        # extra lambda layer to undo this.
        if dimensions == 3:
            expand = lambda var: A.expand_dims(A.expand_dims(A.expand_dims(var, dim=1), dim=1), dim=1)
            expand_shape = lambda _input_shape: (_input_shape[0], 1, 1, 1, _input_shape[1])
        else:
            expand = lambda var: A.expand_dims(A.expand_dims(var, dim=1), dim=1)
            expand_shape = lambda _input_shape: (_input_shape[0], 1, 1, _input_shape[1])
        keras_postprocessing_layer = keras.layers.Lambda(function=expand, output_shape=expand_shape)

    # Build Keras graph
    keras_pooled = keras_poollayer(keras_input)
    keras_output = keras_postprocessing_layer(keras_pooled)
    # Build KerasLayer from keras graph
    layer = KerasLayer(input=keras_input, output=keras_output, name=name)

    layer.window = window
    layer.stride = stride
    layer.border_mode = border_mode
    layer.global_ = global_

    # Done.
    return layer


def upsample(window):
    """Make an upsampling layer with Keras."""
    pass


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
