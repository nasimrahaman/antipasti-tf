__author__ = "Nasim Rahaman"
__doc__ = """
          Antipasti backend. Heavily inspired by the Keras backend, found here:
          https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
          """

import types
from contextlib2 import ExitStack

import numpy as np
import tensorflow as tf


# ------------------- TENSORFLOW-SPECIFIC -------------------

# List of all datatypes
_DATATYPES = ['float16', 'float32', 'float64',
              'int16', 'int32', 'int64', 'uint8', 'unit16',
              'float16_ref', 'float32_ref', 'float64_ref',
              'int16_ref', 'int32_ref', 'int64_ref', 'uint8_ref', 'unit16_ref']

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


def consolidate_context_managers(device=None, variable_scope=None, extra_context_managers=None):
    """Consolidates context managers."""
    extra_context_managers = [] if extra_context_managers is None else extra_context_managers
    more_context_managers = ([tf.device(device)] if device is not None else []) + \
                            ([tf.variable_scope(variable_scope)] if variable_scope is not None else [])
    all_context_managers = more_context_managers + extra_context_managers
    return all_context_managers


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
    return dtype in [dt for dt in _DATATYPES if not dt.endswith('_ref')]


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


# ------------------- VARIABLES-AND-TENSORS -------------------


# Make variable
def variable(value, dtype=_FLOATX, device=None, variable_scope=None, context_managers=None, antipasti_name=None,
             **tf_variable_kwds):
    """
    Makes a tensorflow Variable.

    :type value: numpy.ndarray or float or int
    :param value: Initial value.

    :type dtype: str or Any
    :param dtype: Datatype of the initialized tensor

    :type device: str
    :param device: String specifying where to place the variable.

    :type variable_scope: str
    :param variable_scope: Variable scope to define the variable in.

    :type context_managers: list
    :param context_managers: list of context managers to define the variable in.

    :type tf_variable_kwds: dict
    :param tf_variable_kwds: Dictionary of keyword arguments to send to the tensorflow variable constructor.

    :rtype: tensorflow.Variable
    :return: a tensorflow variable
    """

    # Consolidate context managers
    all_context_managers = consolidate_context_managers(device=device, variable_scope=variable_scope,
                                                        extra_context_managers=context_managers)
    # FIXME Tensorflow can only handle names matching [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (scopes at root)
    # FIXME or [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (other scopes).
    # Set up keyword args for the tf.Variable call
    tf_variable_kwds.update({'initial_value': value})
    with ExitStack() as stack:
        # Enter managers
        for manager in all_context_managers:
            stack.enter_context(manager)
        # Make variable
        var = tf.Variable(dtype=to_tf_dtype(dtype), **tf_variable_kwds)

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


def placeholder(dtype=_FLOATX, shape=None, device=None, variable_scope=None, context_managers=None,
                antipasti_name=None, **tf_placeholder_kwargs):
    """Makes a tensorflow placeholder."""

    # Consolidate context managers
    all_context_managers = consolidate_context_managers(device=device, variable_scope=variable_scope,
                                                        extra_context_managers=context_managers)
    # FIXME Tensorflow can only handle names matching [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (scopes at root)
    # FIXME or [A-Za-z0-9.][A-Za-z0-9_.\\-/]* (other scopes).
    with ExitStack() as stack:
        # Enter all context managers
        for manager in all_context_managers:
            stack.enter_context(manager)
        # Define variable
        ph = tf.placeholder(to_tf_dtype(dtype), shape=shape, **tf_placeholder_kwargs)

    ph._antipasti_name = antipasti_name
    # Return placeholder
    return ph


# ------------------- TENSOR-INFO-AND-MANIPULATION -------------------


def ndim(var):
    """Returns the number of dimensions in a tensor."""
    var_shape = shape(var)
    return None if var_shape is None else len(var_shape)


def shape(var):
    """Returns the shape of a tensor as a python builtin object."""
    _shape = var.get_shape()
    _shape = None if _shape == tf.TensorShape(None) else _shape.as_list()
    return _shape


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


