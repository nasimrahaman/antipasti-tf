__author__ = "Nasim Rahaman"
__doc__ = """
          Antipasti backend. Heavily inspired by the Keras backend, found here:
          https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
          """

import types
import tensorflow as tf


# ------------------- TENSORFLOW-SPECIFIC -------------------

# List of all datatypes
_DATATYPES = ['float16', 'float32', 'float64',
              'int16', 'int32', 'int64', 'uint8', 'unit16',
              'float16_ref', 'float32_ref', 'float64_ref',
              'int16_ref', 'int32_ref', 'int64_ref', 'uint8_ref', 'unit16_ref']

# Default float
_FLOATX = 'float32'


class Session(object):
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


# ------------------- DATATYPE-UTILITIES -------------------


def is_string_dtype(dtype):
    """
    Checks if the given dtype (string) is valid.

    :type dtype: str
    :param dtype: Datatype

    :rtype: bool
    """
    return dtype in _DATATYPES


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


# ------------------- VARIABLES-AND-TENSORS -------------------


# Make variable
def variable(value, dtype=_FLOATX, **tf_variable_kwds):
    """
    Makes a tensorflow Variable.

    :type value: numpy.ndarray
    :param value: Initial value.

    :type dtype: str or Any
    :param dtype: Datatype of the initialized tensor


    :type tf_variable_kwds: dict
    :param tf_variable_kwds: Dictionary of keyword arguments to send to the tensorflow variable constructor.

    :rtype: tensorflow.Variable
    :return: a tensorflow variable
    """
    tf_variable_kwds.update({'initial_value': value})
    var = tf.Variable(dtype=to_tf_dtype(dtype), **tf_variable_kwds)
    # Ah, habits from the good ol' theano days
    var._antipasti_set_value = types.MethodType(set_value, var)
    var._antipasti_get_value = types.MethodType(get_value, var)
    return var


# Set variable value
def set_value(variable, value, session=None):
    pass


def get_value(variable, session=None):
    pass
