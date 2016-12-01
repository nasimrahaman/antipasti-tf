from .. import pyutils as py
from .. import utils


class Model(object):
    """The general model class."""
    def __init__(self, name=None):
        # "Private" variable for name
        self._name = None
        # Set name
        self.name = name

        # Private variables for input and output shapes
        self._input_shape = None
        self._output_shape = None

        # Container for parameters
        self._parameters = utils.ParameterCollection([])

        # Container for input, output and targets
        self._x = None
        self._y = None
        self._yt = None

        # Container for cost and loss (cost = objective = loss + regularization)
        self.C = None
        self.L = None
        # Container for gradients
        self.dC = []
        # Container for updates (which should be assign ops)
        self.updates = []

        # Namespace for storing arbitrary stuff
        self._antipasti_collection = []

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def yt(self):
        return self._yt

    @yt.setter
    def yt(self, value):
        self._yt = value

    @property
    def name(self):
        return str(id(self)) if self._name is None else self._name

    @name.setter
    def name(self, value):
        self._name = value

    def _stamp_string(self, string):
        return "[ModelID:{}] {}".format(self.name, string)

    @property
    def input_shape(self):
        """
        Shape(s) of the layer input tensor(s). If more than one variable go in as inputs, `input_shape` is a list of
        their shapes, ergo a list of lists.

        :rtype list or list of list
        """
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        # Run shape inference
        output_shape = self.infer_output_shape(input_shape=value)
        # If no errors found, set to internal variables
        self._input_shape = value
        self._output_shape = output_shape

    @property
    def output_shape(self):
        """
        Shape(s) of the layer output tensor(s). If more than one variable come out as outputs, `output_shape` is a list
        of their shapes, ergo a list of lists.

        :rtype list or list of list
        """
        return self._output_shape

    @property
    def num_inputs(self):
        """
        Number of inputs to the layer.

        :rtype: int
        """
        # Observe that num_inputs = 1 when self.input_shape = None (legacy behaviour)
        return 1 if not py.islistoflists(self.input_shape) else len(self.input_shape)

    @property
    def num_outputs(self):
        """
        Number of outputs from the layer.

        :rtype: int
        """
        # Observe that num_outputs = 1 when self.input_shape = None (legacy behaviour)
        return 1 if not py.islistoflists(self.output_shape) else len(self.output_shape)

    @property
    def input_dimensions(self):
        """
        Dimensions of the input tensor(s), i.e. `len(input.shape)`.
        If more than one input go in to the layer, `input_dimensions` is a list of input dimensions.

        :rtype: int or list of int
        """
        # Defined only if input_shape is defined.
        if self.input_shape is None:
            raise ValueError(self._stamp_string("Input shape is not known. Set Layer.input_shape first."))
        # Legacy helpers to the rescue!
        return py.delist([len(ishp) for ishp in py.list2listoflists(self.input_shape)])

    @property
    def output_dimensions(self):
        """
        Dimensions of the output tensor(s), i.e. `len(output.shape)`.
        If more than one output comes out from the layer, `output_dimensions` is a list of output dimensions.

        :rtype: int or list of int
        """
        # Defined only if output_shape is defined, which is in turn defined only if input_shape is defined.
        if self.output_shape is None:
            raise ValueError(self._stamp_string("Output shape could not be inferred. Set Layer.input_shape first."))

        return py.delist([len(oshp) for oshp in py.list2listoflists(self.output_shape)])

    @property
    def parameters(self):
        """Parameters (e.g. Weights, Biases, etc.) of the layer."""
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self.assign_parameters(parameters=value)

    def assign_parameters(self, parameters=None):
        raise NotImplementedError

    def infer_output_shape(self, input_shape=None):
        """
        Infer output shape for given `input_shape`.

        :type input_shape: list or list of list
        :param input_shape: Shape of the layer input(s), as a list (1 input) or a list of lists (multiple inputs).
        """
        # This boils down to the default behaviour being to set output_shape = input_shape.
        if input_shape is None:
            input_shape = self.input_shape
        return input_shape

    @utils.forward_pass
    def feedforward(self, input=None):
        """
        Implements the forward pass for the layer, given its input. If the decorator forward_pass is not used, this
        method should read its input from `Model.x` (if necessary) and write its output to `Model.y`.
        """
        return input

    def compute_cost(self):
        pass

    def get_updates(self):
        pass

    def fit(self):
        pass

    def __call__(self, input):
        """This should be remniscent of the Keras functional API."""
        return self.feedforward(input)

    def __add__(self, other):
        """
        Stack depth-wise.

        :type other: Layer or Antipasti.models.tree.LayerTrainyard
        :param other: The other `Layer` or `Model`.
        """
        pass

    def __mul__(self, other):
        """
        Stack width-wise.

        :type other: Layer or Antipasti.models.tree.LayerTrainyard
        :param other: The other `Layer` or `Model`.
        """
        pass


