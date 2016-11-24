from .. import pyutils as py
from .. import utils

from ..layers import Layer

from collections import OrderedDict

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
        self.x = None
        self.y = None
        self.yt = None

        # Container for cost and loss (cost = objective = loss + regularization)
        self.C = None
        self.L = None
        # Container for gradients
        self.dC = []
        # Container for updates (which should be assign ops)
        self.updates = []

        # Namespace for storing arbitrary stuff
        self.collection = []

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
        return 1 if py.islistoflists(self.input_shape) else len(self.input_shape)

    @property
    def num_outputs(self):
        """
        Number of outputs from the layer.

        :rtype: int
        """
        # Observe that num_outputs = 1 when self.input_shape = None (legacy behaviour)
        return 1 if py.islistoflists(self.output_shape) else len(self.output_shape)

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

        :type other: Layer or LayerTrainyard
        :param other: The other `Layer` or `Model`.
        """
        # TODO Can be done once LayerTrainyard is defined.
        pass

    def __mul__(self, other):
        """
        Stack width-wise.

        :type other: Layer or LayerTrainyard
        :param other: The other `Layer` or `Model`.
        """
        # TODO Can be done once LayerTrainyard is defined.
        pass


class LayerTrainyard(Model):
    """Class to implement arbitrary architectures."""
    def __init__(self, trainyard, input_shape=None, name=None):
        """

        :param trainyard:
        :param input_shape:
        """
        # TODO: Write doc
        # Initialze superclass
        super(LayerTrainyard, self).__init__(name=name)

        # Meta
        self._trainyard = None

        # Assign and parse trainyard
        self.trainyard = trainyard

        # Run shape inference
        self.input_shape = input_shape

    @property
    def trainyard(self):
        return self._trainyard

    @trainyard.setter
    def trainyard(self, value):
        # Remove singleton sublists
        value = py.removesingletonsublists(value)
        # Parse value. This is the magic Antipasti line.
        value = [[LayerTrainyard(elem_in_elem) if isinstance(elem_in_elem, list) else elem_in_elem
                  for elem_in_elem in elem]
                 if isinstance(elem, list) else elem for elem in value]
        self._trainyard = value

    @property
    def parameters(self):
        # Note that we don't use self._parameters defined in the superclass.
        return [parameter
                for train in self.trainyard
                for coach in py.obj2list(train)
                for parameter in coach.parameters]

    # Model.input_shape.setter must be overriden to handle e.g. multiple inputs.
    @Model.input_shape.setter
    def input_shape(self, value):
        if value is None:
            # Get input shape from trainyard
            if isinstance(self.trainyard[0], list):
                # Case: Multiple inputs to the trainyard
                _input_shape = [input_layer.input_shape for input_layer in self.trainyard[0]]
            else:
                # Case: Single input to the trainyard
                _input_shape = self.trainyard[0].input_shape
        else:
            # input shape is given.
            _input_shape = value
        # Run shape inference and set _output_shape
        output_shape = self.infer_output_shape(input_shape=_input_shape)
        # Set input and output shapes
        self._input_shape = _input_shape
        self._output_shape = output_shape

    def infer_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape

        # Note that if the trainyard has multiple inputs, input_shape is a list of lists.
        intermediate_shape = input_shape
        # Trains and coaches are legacy terminology from the old theano Antipasti.
        # Loop over all layers or groups of layers (depth-wise)
        for train in self.trainyard:

            if isinstance(train, list):
                # If we're in here, it's because train is a group of width-wise stacked layers
                train_output_shape = []
                # Convert intermediate_shape to a list of lists
                train_input_shape = py.list2listoflists(intermediate_shape)
                # Cursor for keeping track (see below).
                cursor = 0

                # Loop over all layers (width-wise)
                for coach in train:
                    # This will save a function call
                    num_inputs_to_coach = coach.num_inputs
                    # coach can take more than one inputs, and cursor is to keep track of how many inputs have been \
                    # taken from train_input_shape.
                    coach_input_shape = py.delist(train_input_shape[cursor:cursor+num_inputs_to_coach])
                    cursor += num_inputs_to_coach

                    # Assign input shape to coach
                    coach.input_shape = coach_input_shape
                    # Get the resulting output shape and append to the list keeping track of train output shapes
                    train_output_shape.append(coach.output_shape)

            else:
                # If we're in here, it's because train is just a Layer (or a LayerTrainyard). Remember,
                # intermediate_shape can be a list of lists if train takes more than one inputs.
                train.input_shape = intermediate_shape
                train_output_shape = train.output_shape

            # Assign as intermediate shape
            intermediate_shape = py.delist(py.delistlistoflists(py.list2listoflists(train_output_shape)))

        # Done. final_shape = intermediate_shape, but we'll spare us the trouble
        return intermediate_shape

    # Parameter assignment cannot be handled by the superclass
    def assign_parameters(self, parameters=None):
        if parameters is not None:
            # TODO (See Layer.assign_parameters for potential pitfalls)
            pass

    # Feedforward, but without the decorator
    def feedforward(self, input=None):
        # Check if input is given
        if input is None:
            self.x = input
        else:
            input = self.x

        # If trainyard empty: nothing to do, return input
        if not self.trainyard:
            self.y = input
            return self.y

        # The input must be set for all input layers (if there are more than one)
        input_list = py.obj2list(input)
        # The individual input layers may have one or more inputs. The cursor is to keep track.
        cursor = 0

        # Loop over input layers (width-wise)
        for coach in py.obj2list(self.trainyard[0]):
            # Get number of inputs to the coach
            num_inputs_to_coach = coach.num_inputs
            # Fetch from list of inputs
            coach_input = py.delist(input_list[cursor:cursor+num_inputs_to_coach])
            # Increment cursor
            cursor += num_inputs_to_coach
            # Set input
            coach.x = coach_input

        # Feedforward recursively.
        intermediate_result = input
        for train in self.trainyard:
            if isinstance(train, list):
                # Convert intermediate_result to a list
                input_list = py.obj2list(intermediate_result)
                # Make a cursor to index input_list
                cursor = 0
                # List to store outputs from coaches in train
                coach_outputs = []

                for coach in train:
                    num_inputs_to_coach = coach.num_inputs
                    coach_input = py.delist(input_list[cursor:cursor+num_inputs_to_coach])
                    cursor += num_inputs_to_coach
                    # Feedforward and store output in list
                    coach_outputs.append(coach.feedforward(input=coach_input))
                intermediate_result = coach_outputs

            else:
                intermediate_result = train.feedforward(input=intermediate_result)

            # Flatten any recursive outputs to a linear list
            intermediate_result = py.delist(list(py.flatten(intermediate_result)))

        # The final intermediate_result is the final result (no shit sherlock)
        self.y = intermediate_result
        # Done.
        return self.y

    # Depth-wise mechanics
    def __add__(self, other):
        if self.num_outputs != other.num_inputs:
            raise RuntimeError(self._stamp_string("Cannot chain component with {} output(s) with "
                                                  "one with {} inputs.".format(self.num_outputs, other.num_inputs)))
        # Other could be a Layer or a LayerTrainyard
        if isinstance(other, Layer):
            return LayerTrainyard(self.trainyard + [other])
        elif isinstance(other, LayerTrainyard):
            return LayerTrainyard(self.trainyard + other.trainyard)
        else:
            raise TypeError(self._stamp_string("Second summand of invalid type. Expected Layer or LayerTrainyard, "
                                               "got '{}' instead.".format(other.__class__.__name__)))

    # Width-wise mechanics
    def __mul__(self, other):
        # Other could be a Layer or a LayerTrainyard
        if not (isinstance(other, Layer) or isinstance(other, LayerTrainyard)):
            raise TypeError(self._stamp_string("Second summand of invalid type. Expected Layer or LayerTrainyard, "
                                               "got '{}' instead.".format(other.__class__.__name__)))

        return LayerTrainyard([[self, other]])

    # Syntactic candy
    def __getitem__(self, item):
        pass
