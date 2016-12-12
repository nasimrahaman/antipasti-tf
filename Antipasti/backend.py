__author__ = "Nasim Rahaman"
__doc__ = """
          Antipasti backend. Heavily inspired by the Keras backend, found here:
          https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
          """

import re
import types
from argparse import Namespace
from collections import OrderedDict
from functools import partial

import numpy as np
import tensorflow as tf
from contextlib2 import ExitStack, contextmanager

from .legacy import pykit as py
from .utilities.pyutils2 import split_parameter_tag


# ------------------- META -------------------


def get(attr):
    """Get attribute from framework."""
    assert isinstance(attr, str), \
        "Attribute to get must be a string, got {} instead.".format(attr.__class__.__name__)
    return getattr(tf, attr)


def getfw():
    """Get framework."""
    return tf


# ------------------- TENSORFLOW-SPECIFIC -------------------


# List of all datatypes
_DATATYPES = ['float16', 'float32', 'float64',
              'int16', 'int32', 'int64', 'uint8', 'uint16',
              'float16_ref', 'float32_ref', 'float64_ref',
              'int16_ref', 'int32_ref', 'int64_ref', 'uint8_ref', 'uint16_ref']

# Default float
_FLOATX = 'float32'


class TFSession(object):
    """Produces the session used internally by Antipasti."""

    _antipasti_session = None
    _antipasti_session_config = None

    def configure(self, proto):
        """
        Configure a session with a Tensorflow `ConfigProto`.

        :type proto: tensorflow.ConfigProto
        :param proto: Configuration to initialize session with.
        """
        self._antipasti_session_config = proto
        # The following would force the session to reinitialize
        self._antipasti_session = None

    def reset(self):
        """Resets the internal Antipasti Tensorflow Session."""
        self._antipasti_session = None

    @property
    def session(self):
        # If this code is run under a tf.Session() context manager, the default session is set. Make this the session.
        tf_default_session = tf.get_default_session()
        if tf_default_session is not None:
            sess = tf_default_session
        else:
            # Tensorflow has no default session available.
            # Prepare an Antipasti session (if there isn't one already)
            if self._antipasti_session is None:
                # Prepare session
                self._antipasti_session = sess = tf.Session(self._antipasti_session_config)
            else:
                # Antipasti session available
                sess = self._antipasti_session
        return sess

    @session.setter
    def session(self, value):
        self._antipasti_session = value

    def get(self):
        """Get current Tensorflow session."""
        return self.session

    def set(self, value):
        """Set current Tensorflow session."""
        self.session = value


# Define a session
Session = TFSession()


def reinitialize_all_variables(run_init_op=True, session=None):
    """
    Reinitialize all variables and optionally, run the initialization op. Note that already initialized variables
    will also be reinitialized, so handle with care.
    """
    # Get initializer op
    init_op = tf.initialize_all_variables()

    # Run initializer op ...
    if run_init_op:
        # ... with the right session
        session = Session.session if session is None else session
        session.run(init_op)

    return init_op

initialize_all_variables = reinitialize_all_variables


def initialize_all_uninitialized_variables(run_init_op=True, session=None):
    """Initialize only the uninitialized variables."""
    # Get session
    session = Session.session if session is None else session
    # Get list of all uninitialzied variables
    uninitialized_variables = [tf.get_variable(name)
                               for name in tf.report_uninitialized_variables().eval(session=session)]
    # Make init op
    init_op = tf.initialize_variables(uninitialized_variables)
    if run_init_op:
        # Run op
        session.run(init_op)
    # Return init_op for the record
    return init_op


# ------------------- COLLECTION-UTILITIES -------------------


def add_to_collection(name, value):
    tf.add_to_collection(name, value)


def get_from_collection(name, idx=None):
    collection = tf.get_collection(name)
    if idx is None:
        return collection
    else:
        assert isinstance(idx, int), \
            "Index `idx` must be a string, got {} instead.".format(idx.__class__.__name__)
        return collection[idx]


get_collection = partial(get_from_collection, idx=None)


# Collection keys
class Collections(object):
    TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
    WEIGHTS = tf.GraphKeys.WEIGHTS
    BIASES = tf.GraphKeys.BIASES


# ------------------- CONTEXT-MANAGING -------------------


def consolidate_context_managers(device=None, variable_scope=None, extra_context_managers=None):
    """Consolidates context managers."""
    extra_context_managers = [] if extra_context_managers is None else extra_context_managers
    more_context_managers = ([tf.device(device)] if device is not None else []) + \
                            ([tf.variable_scope(variable_scope)] if variable_scope is not None else [])
    all_context_managers = more_context_managers + extra_context_managers
    return all_context_managers


class ContextSupermanager(object):
    """Class to help with managing the usual context managers in tensorflow."""
    def __init__(self, device=None, variable_scope=None, other_context_managers=None):
        """
        :type device: str
        :param device: Device to use, e.g. 'gpu0' or '/gpu:0'

        :type variable_scope: str or list of str or any
        :param variable_scope: (List of) variable scopes. If strings are provided, they're wrapped in
                               tf.variable_scope

        :type other_context_managers: list
        :param other_context_managers: List of extra context managers to be used.
        """
        # Book keeping
        self._device = None
        self._scope_yields = None
        self._variable_scope = None
        self._other_context_managers = None

        # Attach meta
        self.device = device
        self.variable_scope = variable_scope
        self.other_context_managers = other_context_managers

    def get_managers(self, parameter_tag=None, layer_id=None, device=None, variable_scope=None,
                     other_context_managers=None, reuse=None, reuse_variable_scope=None,
                     reuse_layer_variable_scope=None):
        """
        :type parameter_tag: str or NoneType
        :param parameter_tag: Parameter tag of the layer for which the manager is being retrieved.

        :type layer_id: str or NoneType
        :param layer_id: ID of the layer for which the manager is being retrieved.

        :type device: str or NoneType
        :param device: Device to use, e.g. 'gpu0' or '/gpu:0'

        :type variable_scope: str or list of str or NoneType or any
        :param variable_scope: (List of) variable scopes. If strings are provided, they're wrapped in
                               tf.variable_scope

        :type other_context_managers: list or NoneType
        :param other_context_managers: List of extra context managers to be used.

        :type reuse: bool or NoneType
        :param reuse: Whether to reuse variables in all variable scope. Note that this argument takes precedence over
                      `reuse_variable_scope` and `reuse_layer_variable_scope`.

        :type reuse_variable_scope: bool or NoneType
        :param reuse_variable_scope: Whether to reuse variable scopes in the `variable_scope` argument.

        :type reuse_layer_variable_scope: bool or NoneType
        :param reuse_layer_variable_scope: Whether to reuse the variable scope of the layer
                                           (as deduced from `parameter_tag` or `layer_id`)

        :return: List of context mangers, ready to be entered in an ExitStack
        """
        # Get device
        device = self.device if device is None else self.parse_device_name(device)
        device = [tf.device(device)]

        # Figure out which variable scopes are to be set reusable. The 'reuse' argument takes precedence
        if reuse is not None:
            reuse_variable_scope = reuse_layer_variable_scope = reuse

        # Get variable scope from parameter tag
        if parameter_tag is not None:
            assert isinstance(parameter_tag, str), \
                "`parameter_tag` must be a string, got {} instead.".format(parameter_tag.__class__.__name__)
            layer_id_from_tag, _ = split_parameter_tag(parameter_tag, check=True)
            assert (layer_id is None) or layer_id_from_tag == layer_id, \
                "Provided layer_id {} is not consistent with " \
                "that obtained from the parameter tag {} ({}).".format(layer_id, parameter_tag,
                                                                       layer_id_from_tag)
            layer_id = layer_id_from_tag

        # Get variable scope from the layer_id known so far (if at all)
        if layer_id is not None:
            assert isinstance(layer_id, str), \
                "`layer_id` must be a string, got {} instead.".format(layer_id.__class__.__name__)
            # Make variable scope with layer_id
            layer_variable_scope = [tf.variable_scope('layer-id_{}'.format(layer_id),
                                                      reuse=reuse_layer_variable_scope)]
        else:
            # Alright then, no variable scope for this layer specified
            layer_variable_scope = []

        # Get extra variable scopes if any provided
        if variable_scope is None:
            # Read scope from attribute
            variable_scope = self.variable_scope if self.variable_scope is not None else []

        # Support for variable_scope passed as a string or a list of strings
        variable_scope = [tf.variable_scope(scope, reuse=reuse_variable_scope) if isinstance(scope, str) else scope
                          for scope in py.obj2list(variable_scope)]

        # Get the remaining context managers
        other_context_managers = py.obj2list(other_context_managers) if other_context_managers is not None else []

        # Build a list of all context managers and return.
        all_context_managers = OrderedDict([('device_scope', device),
                                            ('layer_variable_scope', layer_variable_scope),
                                            ('variable_scope', variable_scope),
                                            ('other_context_managers', other_context_managers)])

        # Return
        return all_context_managers

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = self.parse_device_name(value)

    @property
    def variable_scope(self):
        return self._variable_scope

    @variable_scope.setter
    def variable_scope(self, value):
        # Parse value (convert to list)
        if value is None:
            value = []
        else:
            value = py.obj2list(value)
        # Done.
        self._variable_scope = value

    @property
    def other_context_managers(self):
        return self._other_context_managers

    @other_context_managers.setter
    def other_context_managers(self, value):
        # Parse value (convert to list)
        if value is None:
            value = []
        else:
            value = py.obj2list(value)
        # Done.
        self._other_context_managers = value

    @property
    def scope_yields(self):
        return Namespace(**self._scope_yields)

    @staticmethod
    def parameter_tag_to_variable_scope(parameter_tag):
        if parameter_tag is not None:
            layer_id, parameter_name = split_parameter_tag(parameter_tag, check=True)
            return layer_id
        else:
            return None

    @contextmanager
    def manage(self, parameter_tag=None, layer_id=None, device=None, variable_scope=None,
               other_context_managers=None, reuse=None, reuse_layer_variable_scope=None,
               reuse_variable_scope=None):

        with ExitStack() as stack:
            # We're gonna store all mangers yields in an ordered dict, which will in turn be (indirectly) yielded by
            # this manager (indirectly because this manager yields self for full access, and the manager_yields
            # ordered dict is assigned as the attribute 'scope_yields').
            _manager_yields = OrderedDict([])
            for manager_group, managers in self.get_managers(parameter_tag=parameter_tag, layer_id=layer_id,
                                                             device=device, variable_scope=variable_scope,
                                                             other_context_managers=other_context_managers,
                                                             reuse=reuse,
                                                             reuse_layer_variable_scope=reuse_layer_variable_scope,
                                                             reuse_variable_scope=reuse_variable_scope).items():
                _manager_yields[manager_group] = []
                for manager in managers:
                    _manager_yield = stack.enter_context(manager)
                    _manager_yields[manager_group].append(_manager_yield)

            self._scope_yields = _manager_yields
            # Yield the object back
            yield self

    def reuse_variables(self, in_layer_variable_scope=True, in_other_variable_scopes=True):
        """
        Whether to reuse variables in the variable scopes. For documentation on how variable scopes work, see:
        https://www.tensorflow.org/versions/master/how_tos/variable_scope/index.html

        :type in_layer_variable_scope: bool
        :param in_layer_variable_scope: Whether to reuse variables in the variable scopes used by the layer
                                        (if any defined by providing a parameter_tag or a layer_id).

        :type in_other_variable_scopes: bool
        :param in_other_variable_scopes: Whether to reuse variables in the user-defined variable scopes
                                         (if any defined).
        """
        if in_layer_variable_scope:
            for scope_yield in self.scope_yields.layer_variable_scope:
                scope_yield.reuse_variables()

        if in_other_variable_scopes:
            for scope_yield in self.scope_yields.variable_scope:
                scope_yield.reuse_variables()

    @staticmethod
    def parse_device_name(device):
        if device is None:
            return ''
        elif re.match('[g, c]pu[0-9]*', device):
            return '/{}:{}'.format(device[0:3], device[3:] if device[3:] else '0')
        else:
            # Failed to parse; maybe it's an address in distributed tf or already in the right format
            return device


def call_in_managers(context_managers=None):
    """
    Decorator factory that makes a decorator to call the decorated function within nested `context_managers`.

    :type context_managers: list
    :param context_managers: List of context managers to nest over. The first manager in list is entered first.
    """
    def _decorator(function):
        def decorated_function(*args, **kwargs):
            with ExitStack() as stack:
                # Enter managers
                for manager in context_managers:
                    stack.enter_context(manager)
                # Evaluate function
                output = function(*args, **kwargs)
            return output
        return decorated_function
    return _decorator


# ------------------- DATATYPE-UTILITIES -------------------


def is_string_dtype(dtype):
    """
    Checks if the given dtype (string) is valid.

    :type dtype: str
    :param dtype: Datatype

    :rtype: bool
    """
    # Beware that e.g. tf.float32 == 'float32', so we need the extra isinstance check in place.
    return isinstance(dtype, str) and dtype in [dt for dt in _DATATYPES if not dt.endswith('_ref')]


def is_tf_dtype(dtype):
    """
    Checks if the given dtype (tf.[datatype]) is valid.

    :rtype: bool
    """
    return dtype in [getattr(tf, dt) for dt in _DATATYPES]


def to_tf_dtype(dtype):
    """Convert given datatype `dtype` to tensorflow.dtype if it isn't one already."""
    if not is_string_dtype(dtype):
        # Check if it's a tensorflow data type
        if not is_tf_dtype(dtype):
            raise ValueError("Datatype {} is not supported.".format(dtype))
        else:
            # If it indeed is a tensorflow datatype (passed by a forgivable mistake), return
            return dtype
    return getattr(tf, dtype)


def unref_tf_dtype(dtype):
    """Converts e.g. tf.float32_ref to tf.float32."""
    # Make sure dtype is a tf.dtype
    dtype = to_tf_dtype(dtype)
    # Check if '_ref' in name
    if dtype.name.endswith('_ref'):
        dtype_str = dtype.name[:-4]
        return to_tf_dtype(dtype_str)
    else:
        return dtype


def to_tf_tensor(value, dtype=_FLOATX, name=None):
    return tf.convert_to_tensor(value, dtype=dtype, name=name)


# ------------------- VARIABLES-AND-TENSORS -------------------


# Make variable
def variable(value=None, name=None, shape=None, dtype=_FLOATX, context_supermanager=None,
             device=None, variable_scope=None, other_context_managers=None, antipasti_name=None,
             **tf_variable_kwds):
    """
    Makes or gets a tensorflow Variable. If value is not provided but name is (and optionally, variable_scope names),
    this function calls the tf.get_variable method. In this case, the shape must be provided if the variable is being
    initialized for the first time. Otherwise if a value is provided, this function calls the
    tf.Variable constructor directly. If both value and name are provided, value takes precedence.

    :type value: numpy.ndarray or float or int
    :param value: Initial value.

    :type name: str
    :param name: Tensorflow name.

    :type shape: list
    :param shape: Shape of the variable.

    :type dtype: str or Any
    :param dtype: Datatype of the initialized tensor

    :type context_supermanager: ContextSupermanager
    :param context_supermanager: A context supermanager to initialize the variable in.

    :type device: str
    :param device: String specifying where to place the variable.

    :type variable_scope: str
    :param variable_scope: Variable scope to define the variable in.

    :type other_context_managers: list
    :param other_context_managers: list of context managers to define the variable in.

    :type antipasti_name: str
    :param antipasti_name: Variable name to be used by Antipasti.

    :type tf_variable_kwds: dict
    :param tf_variable_kwds: Dictionary of keyword arguments to send to the tensorflow variable constructor.

    :rtype: tensorflow.Variable
    :return: a tensorflow variable
    """

    # Make context supermanager if none provided
    if context_supermanager is None:
        context_supermanager = ContextSupermanager(device=device, variable_scope=variable_scope,
                                                   other_context_managers=other_context_managers)

    # Check whether to get or to make
    make = value is not None
    get = name is not None

    with context_supermanager.manage():
        if make:
            # Set up keyword args for the tf.Variable call
            tf_variable_kwds.update({'initial_value': to_tf_tensor(value, dtype=dtype),
                                     'name': name})
            # Make variable
            var = tf.Variable(dtype=to_tf_dtype(dtype), **tf_variable_kwds)
        elif get:
            # Get variable from scope
            var = tf.get_variable(name, shape=shape, dtype=dtype, **tf_variable_kwds)
        else:
            raise RuntimeError("Either value or name must be provided.")

    # Ah, habits from the good ol' theano days
    var._antipasti_set_value = types.MethodType(set_value, var)
    var._antipasti_get_value = types.MethodType(get_value, var)
    var._antipasti_collection = {}
    var._antipasti_name = antipasti_name
    return var


def set_value(var, value, session=None):
    """
    Set variable value. Also available as an attribute (to variable) if the variable was created with the `variable`
    function (in scope).
    """
    # Make sure value is an array
    value = np.asarray(value)
    # Get variable data type
    dtype = unref_tf_dtype(var.dtype)

    # Check if assign_placeholder and op are defined
    if var._antipasti_collection.get('assign_placeholder') is None:
        _placeholder = var._antipasti_collection['assign_placeholder'] = tf.placeholder(dtype, shape=value.shape)
        var._antipasti_collection['assign_op'] = var.assign(_placeholder)

    # Figure out which session to use
    if session is None:
        session = Session.session

    # Run assign op
    session.run(var._antipasti_collection['assign_op'],
                feed_dict={var._antipasti_collection['assign_placeholder']: value})


def get_value(var, session=None):
    """
    Get variable value. Also available as an attribute (to variable) if the variable was created with the `variable`
    function (in scope).
    """
    return var.eval(session=(session if session is not None else Session.session))


def placeholder(dtype=_FLOATX, shape=None, context_supermanager=None, device=None, variable_scope=None,
                other_context_managers=None, antipasti_name=None, **tf_placeholder_kwargs):
    """Makes a tensorflow placeholder."""

    # Build context supermanager
    if context_supermanager is None:
        context_supermanager = ContextSupermanager(device=device, variable_scope=variable_scope,
                                                   other_context_managers=other_context_managers)
    # Manage contexts and define placeholder
    with context_supermanager.manage():
        # Define variable
        ph = tf.placeholder(to_tf_dtype(dtype), shape=shape, **tf_placeholder_kwargs)

    ph._antipasti_name = antipasti_name
    # Return placeholder
    return ph


# ------------------- TENSOR-INFO-AND-MANIPULATION -------------------


def ndim(tensor):
    """Returns the number of dimensions in a tensor."""
    var_shape = shape(tensor)
    return None if var_shape is None else len(var_shape)


def shape(tensor, symbolic=False):
    """Returns the shape of a tensor, by default as a non-symbolic object."""
    if not symbolic:
        _shape = tensor.get_shape()
        _shape = None if _shape == tf.TensorShape(None) else _shape.as_list()
        return _shape
    else:
        return tf.shape(tensor)


def concatenate(tensors, axis=0, name='concat'):
    if axis < 0:
        # We need to determine the number of dimensions in the tensor because tensorflow can't (as of the day)
        _ndims = [ndim(tensor) for tensor in tensors]
        # Check if the number of dimensionsions can be computed
        if all([_ndim is None for _ndim in _ndims]):
            raise ValueError("Number of dimensions could not be computed "
                             "for concatenation along axis {}.".format(axis))

        # ... and whether it's consistent
        all_num_dimensions = filter(lambda x: x is not None, _ndims)
        if not all([_ndim == all_num_dimensions[0] for _ndim in all_num_dimensions]):
            raise ValueError("Can only concatenate tensors with the same number of dimensions.")

        # Get number of dimensions and compute the axis
        num_dimension = max(all_num_dimensions)
        axis = axis % num_dimension
    # Finally,
    return tf.concat(axis, tensors, name=name)


def add_n(tensors, name='add_n'):
    return tf.add_n(inputs=tensors, name=name)


def expand_dims(tensor, dim, name='expand_dims'):
    return tf.expand_dims(tensor, dim, name=name)


def transpose(tensor, perm=None, name='transpose'):
    return tf.transpose(tensor, perm=perm, name=name)


def reshape(tensor, shape, name='reshape'):
    return tf.reshape(tensor, shape=shape, name=name)
