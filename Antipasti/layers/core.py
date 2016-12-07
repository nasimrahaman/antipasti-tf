__author__ = "Nasim Rahaman"

from ..utilities import utils
from .. import backend as A
from ..legacy import pykit as py
from ..models.tree import LayerTrainyard
from ..utilities import pyutils2 as py2


class Layer(object):
    """
    Abstract Layer class. This class implements basic layer mechanics (addition and multiplication) in addition to
    parameter value assignment.
    """

    # WARNING: Renaming 'Layer' would break Antipasti.models.tree.LayerTrainyard.__add__ and
    # Antipasti.models.tree.LayerTrainyard.__mul__. Be sure to make the necessary changes there.
    def __init__(self, name=None, context_supermanagers=None, device=None, variable_scope=None,
                 other_context_managers=None):
        """
        Constructor for the Layer superclass.

        :type name: str
        :param name: Layer name (optional)

        :type device: str
        :param device: Device to place the layer on.

        :type variable_scope: str
        :param variable_scope: Name of the tensorflow varaible scope

        :type other_context_managers: list
        :param other_context_managers: List of context managers within which this layer is to be built.
        """

        # "Private" variable for name
        self._name = None
        # Set name
        self.name = name

        # Set context supermanager
        self.layer_context_supermanagers = context_supermanagers if context_supermanagers is not None else \
            utils.get_layer_context_supermanagers(device=device, variable_scope=variable_scope,
                                                  other_context_managers=other_context_managers)

        # "Private" variables for input and output shapes
        self._input_shape = None
        self._output_shape = None

        # Container for parameters
        self._parameters = py2.ParameterCollection([])

        # Containers for input and output
        self._x = None
        self._y = None

        # Flag to set if a layer has been feedforward
        self._is_fedforward = False
        self._is_initialized = False

        # A namespace for storing arbitrary stuff (implemented as a dict for its `get` method)
        self._antipasti_collection = {}

    @property
    def x(self):
        # This function does the following:
        # 1. Checks if _x has already been defined; if not, it defines it
        # 2. Checks if the input_shape has changed since _x was last defined, in which case it redefines it.

        # Check if the input shape has changed since _x was last defined (remember that _x can also be a list).
        _xs = py.obj2list(self._x)

        # Make double sure _xs is a list of the right length
        assert len(_xs) == 1 or len(_xs) == self.num_inputs, \
            self._stamp_string("Internal inconsistency: was expecting self._x to have 1 or {} "
                               "elements, got {} inputs instead.".format(self.num_inputs, len(_xs)))

        # A layer with multiple inputs will still have [None] in _xs if not for the following line
        _xs = _xs * self.num_inputs if len(_xs) != self.num_inputs else _xs
        _input_shapes = py.list2listoflists(self.input_shape)
        # Loop over all inputs to the layer
        for _x_num in range(self.num_inputs):
            # Fetch current input
            _x = _xs[_x_num]
            # Get input shape if possible
            _x_shape = A.shape(_x) if _x is not None else None
            # Reinitialize variable if it's not initialized yet or if the shapes are out of whack.
            # To start, we check if a reinitialization is required
            # (it's required if the variable is not set yet, and if a soft shape comparison (i.e. ignoring None) fails).
            reinitialization_required = _xs[_x_num] is None or \
                                        not utils.compare_shapes(_x_shape, _input_shapes[_x_num], soft=True)
            # If a reinit is not required, check if some shape setting is in order, i.e. when soft check passes but the
            # hard check fails
            shape_setting_required = not reinitialization_required and \
                                     utils.compare_shapes(_x_shape, _input_shapes[_x_num], soft=True) and \
                                     not utils.compare_shapes(_x_shape, _input_shapes[_x_num], soft=False)

            if reinitialization_required:
                _xs[_x_num] = utils.get_layer_xy_placeholders(input_shape=_input_shapes[_x_num],
                                                              device=self.device,
                                                              variable_scope=self.variable_scope,
                                                              layer_id=self.name,
                                                              context_managers=self.given_context_managers)['x']
                self._is_fedforward = False
            elif shape_setting_required:
                # We're almost good, just need to set the shape
                _xs[_x_num].set_shape(_input_shapes[_x_num])
            else:
                # Nothing to do, we're good
                pass

        # Unwrap xs and set as new _x
        self._x = py.delist(_xs)
        # Return
        return self._x

    @x.setter
    def x(self, value):
        # Convert everything to list so we could loop over
        values = py.obj2list(value)
        _xs = py.obj2list(self._x)
        _input_shapes = py.list2listoflists(self.input_shape)
        for _x_num in range(self.num_inputs):
            value_shape = A.shape(values[_x_num])
            if value_shape is None or utils.compare_shapes(value_shape, _input_shapes[_x_num], soft=True):
                # Shapes are ok
                _xs[_x_num] = values[_x_num]
            else:
                raise ValueError(self._stamp_string("The {}-th input's shape is unexpected "
                                                    "(was expecting {}, found {})".
                                                    format(_x_num, _input_shapes[_x_num], value_shape)))

        self._x = py.delist(_xs)

    @property
    def y(self):
        # Check if y has been defined
        if self._y is not None and self._is_fedforward:
            # Validate shape
            if not utils.validate_shape(self._y, self.output_shape, set_shape=True):
                raise RuntimeError(self._stamp_string("Internal inconsistency: was expecting "
                                                      "self._y to have shape {}, got {} instead.".
                                                      format(utils.get_shape(self._y), self.output_shape)))
            return self._y
        else:
            raise RuntimeError(self._stamp_string("The output (y) is not defined yet. Consider calling "
                                                  "the feedforward method before trying to access this variable."))

    @y.setter
    def y(self, value):
        if not py.smartlen(value) == self.num_outputs:
            raise ValueError(self._stamp_string("Expected {} outputs (y), got {}.".
                                                format(self.num_outputs, py.smartlen(value))))

        # The following checks if value shape is consistent:
        # Make sure _ys has the right length
        _ys = py.obj2list(self._y)
        assert len(_ys) == 1 or len(_ys) == self.num_outputs, \
            self._stamp_string("Internal Error: Was expecting 1 or {} elements "
                               "in self._y, found {}.".format(self.num_outputs, len(_ys)))
        _ys = _ys * self.num_outputs if len(_ys) == 1 else _ys

        # value.shape could be None if it's not set, so we can't use utils.get_shape
        value_shapes = [A.shape(val) for val in py.obj2list(value)]

        for _y_num in range(self.num_outputs):
            # Figure out whether to check shapes, i.e. only when value_shape is given and self._y is not None,
            # which must not always be the case.
            check_shapes = value_shapes[_y_num] is not None and _ys[_y_num] is not None
            if check_shapes and not utils.validate_shape(_ys[_y_num], value_shapes[_y_num], set_shape=True):
                raise ValueError(self._stamp_string("Trying to set input variable {} with "
                                                    "expected shape {} to a value of shape {}.".
                                                    format(_y_num, A.shape(_ys[_y_num]), value_shapes[_y_num])))

        self._y = value

    @property
    def name(self):
        return str(id(self)) if self._name is None else self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def context_managers(self):
        # Consolidate context managers
        return self.layer_context_supermanagers.manage()

    @property
    def device(self):
        return self.layer_context_supermanagers.device

    @device.setter
    def device(self, value):
        self.layer_context_supermanagers.device = value

    @property
    def variable_scope(self):
        return self.layer_context_supermanagers.variable_scope

    @variable_scope.setter
    def variable_scope(self, value):
        self.layer_context_supermanagers.variable_scope = value

    @property
    def other_context_managers(self):
        return self.layer_context_supermanagers.other_context_managers

    @other_context_managers.setter
    def other_context_managers(self, value):
        self.layer_context_supermanagers.other_context_managers = value

    def _stamp_string(self, string):
        return "[LayerID:{}] {}".format(self.name, string)

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
        self._input_shape = py.delistlistoflists(value)
        self._output_shape = py.delistlistoflists(output_shape)

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

    def register_parameter(self, variable, name=None):
        # TODO: (1) register flags in _antipasti_collection ...
        # ... (e.g. trainable, regularizable, regularization coefficient, max_norm, etc.)
        # Get variable name from variable if name is not given
        name = name if name is not None else variable.name
        # Write to dict
        self._parameters['[LayerID:{}][{}]'.format(self.name, name)] = variable
        # TODO Add to GraphKeys.TRAINABLE_VARIABLES
        return variable

    def initialize_and_register_parameter(self, shape, initialization, name=None):
        # TODO
        pass

    @utils.shape_inference
    def infer_output_shape(self, input_shape=None):
        """
        Infer output shape for given `input_shape`. The decorator utils.shape_inference guarantees that the
        input_shape argument is given.

        :type input_shape: list or list of list
        :param input_shape: Shape of the layer input(s), as a list (1 input) or a list of lists (multiple inputs).
        """
        # This boils down to the default behaviour being to set output_shape = input_shape.
        if input_shape is None:
            input_shape = self.input_shape
        return input_shape

    @utils.layer_initialization
    def initialize_layer(self, input_shape=None):
        """
        Define parameters to be used by the layers in this function for given `input_shape`.
        The decorator utils.layer_initialization guarantees that the input_shape is given, the variables are
        initialized in the right device and under the right context managers (including variable_scopes), and that the
        self._is_initialized flag is set to True.

        :type input_shape: list or list of list
        :param input_shape: Shape of the layer input(s), as a list (1 input) or a list of lists (multiple inputs).
        """
        # Define parameters here. Register parameters with the Layer.register_parameter method.
        pass

    @utils.forward_pass
    def feedforward(self, input=None):
        """
        Implements the forward pass for the layer, given its input. If the decorator forward_pass is not used, this
        method should read its input from `Layer.x` (if necessary) and write its output to `Layer.y`.
        """
        # input is by default None, but thanks to the utils.forward_pass decorator, it will always get an argument.
        return input

    def assign_parameters(self, parameters=None):
        """
        Given a list of parameters (numpy arrays or tf.Variable), assign them as layer parameters.

        :type parameters: list
        :param parameters: List of parameters (as numpy array or tf.Variable).
        """
        if parameters is not None:
            # Parameter assignment will happen with the default Antipasti session
            for parameter_variable, parameter_value in zip(self.parameters, parameters):
                A.set_value(var=parameter_variable, value=parameter_value)

    def __call__(self, input):
        """This should be remniscent of the Keras functional API."""
        return self.feedforward(input)

    def __add__(self, other):
        """
        Stack depth-wise.

        :type other: Layer or LayerTrainyard
        :param other: The other `Layer` or `LayerTrainyard`.
        """

        if self.num_outputs != other.num_inputs:
            raise RuntimeError(self._stamp_string("Cannot chain component with {} output(s) with "
                                                  "one with {} inputs.".format(self.num_outputs, other.num_inputs)))
        # Other could be a Layer or a LayerTrainyard
        if isinstance(other, Layer):
            return LayerTrainyard([self, other])
        elif isinstance(other, LayerTrainyard):
            return LayerTrainyard([self] + other.trainyard)
        else:
            raise TypeError(self._stamp_string("Second summand of invalid type. Expected Layer or LayerTrainyard, "
                                               "got '{}' instead.".format(other.__class__.__name__)))

    def __mul__(self, other):
        """
        Stack width-wise.

        :type other: Layer or LayerTrainyard
        :param other: The other `Layer` or `LayerTrainyard`.
        """
        # Other could be a Layer or a LayerTrainyard
        if not (isinstance(other, Layer) or isinstance(other, LayerTrainyard)):
            raise TypeError(self._stamp_string("Second summand of invalid type. Expected Layer or LayerTrainyard, "
                                               "got '{}' instead.".format(other.__class__.__name__)))

        return LayerTrainyard([[self, other]])
