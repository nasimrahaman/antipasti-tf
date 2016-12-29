__author__ = "Nasim Rahaman"

from warnings import warn
from ..utilities import utils
from ..utilities import pyutils2 as py2
from ..legacy import pykit as py
from .. import backend as A


class TrainingConfiguration(object):
    """Configuration class for an Antipasti Trainer."""
    # TODO
    pass


class Trainer(object):
    """Abstract class for an Antipasti Trainer."""
    # TODO
    pass


def application(method):
    """Decorator for the apply functions. Fetches model from the class (if None provided) and applies the method."""

    def _method(cls, model=None):

        if model is None:
            model = cls.model
        else:
            cls.model = model

        if model is None:
            raise RuntimeError(cls._stamp_string("Cannot apply {}, no model is provided.".
                                               format(cls.__class__.__name__)))

        method(cls, model=model)
        return model

    return _method


class ModelApp(object):

    _ALLOWED_KWARGS = None

    def _stamp_string(self, string):
        if hasattr(self, 'model') and self.model is not None:
            return self.model._stamp_string(string)
        else:
            return "[{}:{}] {}".format(self.__class__.__name__, id(self), string)

    def _validate_kwargs(self, **kwargs):
        if self._ALLOWED_KWARGS is None:
            return
        # Make sure all keywords are in the set of allowed keywords
        for key in kwargs.keys():
            if key not in self._ALLOWED_KWARGS:
                raise KeyError(self._stamp_string("The given keyword {} is not expected. "
                                                  "Allowed keywords are: {}.".
                                                  format(key, self._ALLOWED_KWARGS)))

    def _reset_attributes(self, reset_what, name_attribute_map):
        # Given a name_attribute_map and a iterable of whats,
        # this function does the resetting.
        for what in reset_what:
            assert what in name_attribute_map.keys(), \
                self._stamp_string("Unexpected reset key: '{}'. Allowed keys are: {}.".
                                   format(what, name_attribute_map.keys()))
        if not reset_what:
            # Get rid of every attribute
            reset_what = name_attribute_map.keys()

        for what in reset_what:
            name_attribute_map[what] = None

    def reset(self, *reset_what):
        raise NotImplementedError

    @property
    def model_is_bound(self):
        return hasattr(self, 'model') and getattr(self, 'model') is not None

    def unbind_model(self):
        """Gets rid of the bound model instance."""
        if hasattr(self, 'model'):
            # noinspection PyAttributeOutsideInit
            self.model = None

    def attach_to_model_without_binding(self, model):
        raise NotImplementedError

    @application
    def apply(self, model):
        raise NotImplementedError


class Optimizer(ModelApp):
    """Abstract class for an optimizer."""
    def __init__(self, model=None, **kwargs):
        self.model = model

    @application
    def apply(self, model=None):
        """Get the training op and write as model attribute."""
        return model


class Loss(ModelApp):
    """Abstract class for a loss function (not neccesarily an objective)."""

    _ALLOWED_KWARGS = {'aggregation_method', 'weights', 'y', 'yt', 'loss_vector', 'loss_scalar', 'method'}

    def __init__(self, model=None, **kwargs):
        # Property containers
        self._model = None
        self._weights = None
        self._aggregation_method = None
        self._method = None
        self._loss_vector = None
        self._loss_scalar = None
        self._y = None
        self._yt = None

        # Set model
        self.model = model

        # Validate kwargs
        self._validate_kwargs(**kwargs)

        # Read from kwargs
        self.aggregation_method = kwargs.get('aggregation_method', default='mean')
        self.method = kwargs.get('method')
        self.weights = kwargs.get('weights')
        self.y = kwargs.get('y')
        self.yt = kwargs.get('yt')
        self.loss_vector = kwargs.get('loss_vector')
        self.loss_scalar = kwargs.get('loss_scalar')

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        # Clear loss_vector and loss_scalar
        self.reset('loss_vector', 'loss_scalar')

    @property
    def weights(self):
        """Weights for weighing the objective."""
        return self._weights

    @weights.setter
    def weights(self, value):
        # Do nothing if value is None
        if value is None:
            return
        # value must have the same shape as y (prediction) and yt (target) in all but
        # the channel axis, where it must be of size 1.
        value_shape = A.shape(value)
        if value_shape is not None:
            # Check whether shape is right along the channel axis
            assert value_shape[-1] == 1 or value_shape[-1] is None, \
                self._stamp_string("The weight tensor of inferred shape {} must be of length 1 along "
                                   "the channel axis, got {} instead.".format(value_shape, value_shape[-1]))
        self._weights = value

    @property
    def aggregation_method(self):
        return self._aggregation_method

    @aggregation_method.setter
    def aggregation_method(self, value):
        assert value in {'mean', 'sum'}, \
            self._stamp_string("Aggregation method must be "
                               "in ['mean', 'sum']. Got {} instead.".
                               format(value))
        self._aggregation_method = value
        self.reset('loss_scalar')

    @property
    def method(self):
        """Method. Should be a callable."""
        if self._method is not None:
            return self._method
        else:
            raise RuntimeError(self._stamp_string("Loss `method` is yet to be set."))

    @method.setter
    def method(self, value):
        # Do nothing if value is None
        if value is None:
            return
        self._set_method(value)
        self.reset('loss_vector', 'loss_scalar')

    def _set_method(self, method, is_keras_objective=False):
        # TODO: Fetch from registry if value is a string
        pass

    @property
    def y(self):
        # If this instance is bound to a model, we use the y defined in the model.
        if self.model is not None:
            return self.model.y
        elif self._y is None:
            # No model is provided. We make sure self._y is not None.
            raise RuntimeError(self._stamp_string("`y` (prediction) is yet to be defined."))
        else:
            return self._y

    @y.setter
    def y(self, value):
        if self.model is not None:
            warn(self._stamp_string("Setting `y` has no effect if the model is bound."
                                    "To unbind the model, call the `unbind_model` method first."),
                 RuntimeWarning)
        else:
            self.reset('loss_vector', 'loss_scalar')
            self._y = value

    @property
    def yt(self):
        # If this instance is bound to a model, we use the yt defined in the model.
        if self.model is not None:
            return self.model.yt
        elif self._yt is None:
            # No model is provided. We make sure self._yt is not None.
            raise RuntimeError(self._stamp_string("`yt` (target) is yet to be defined."))
        else:
            return self._yt

    @yt.setter
    def yt(self, value):
        # If model is bound, set model target
        if self.model is not None:
            # We let model `yt` setter do the validation
            self.model.yt = value
        else:
            # No model is provided
            # loss_vector and loss_scalar are now to be recomputed, so we get rid of the
            # cached tensors
            self.reset('loss_vector', 'loss_scalar')
            self._yt = value

    def assert_y_and_yt_shapes_are_compatible(self):
        y_shape = A.shape(self.y)
        yt_shape = A.shape(self.yt)
        assert utils.compare_shapes(y_shape, yt_shape), \
            self._stamp_string("Shape of prediction `y` (= {}) is not compatible "
                               "with that of target `yt` (= {}).".format(y_shape, yt_shape))

    @property
    def _y_is_defined(self):
        return (self.model is not None) or (self._y is not None)

    @property
    def _yt_is_defined(self):
        return (self.model is not None) or (self._y is not None)

    @property
    def loss_vector(self):
        # Check if loss vector is already cached
        if self._loss_vector is not None:
            return self._loss_vector
        else:
            # if not, validate y and yt
            self.assert_y_and_yt_shapes_are_compatible()
            # ... and compute loss vector (symbolically)
            self._loss_vector = self._get_loss_vector()
            return self._loss_vector

    @loss_vector.setter
    def loss_vector(self, value):
        # Do nothing else if value is None
        if value is None:
            return
        # Validate value ndim if possible
        value_ndim = A.ndim(value)
        if value_ndim is not None:
            value_shape = A.shape(value)
            assert value_ndim == 1, self._stamp_string("Expected loss vector to be a vector (1-D tensor), "
                                                       "got a {}-D tensor of shape {} instead.".
                                                       format(value_ndim, value_shape))
        self._loss_vector = value
        # A new loss_scalar needs to be computed, so we get rid of the one in cache
        self.reset('loss_scalar')

    def _get_loss_vector(self):
        """Get loss as a vector, such that its length equals the number of batches."""
        # Flatten
        flattened_y = A.image_tensor_to_matrix(self.y)
        flattened_yt = A.image_tensor_to_matrix(self.yt)
        flattened_weights = A.image_tensor_to_matrix(self.weights)
        # Evaluate loss and weight
        unweighted_loss_vector = self.method(flattened_y, flattened_yt)
        weighted_loss_vector = self.apply_weights(unweighted_loss_vector, flattened_weights)
        # Done.
        return weighted_loss_vector

    @property
    def loss_scalar(self):
        # Check if loss scalar is already cached
        if self._loss_scalar is not None:
            return self._loss_scalar
        else:
            self._loss_scalar = self._get_loss_scalar()

    @loss_scalar.setter
    def loss_scalar(self, value):
        # Do nothing else if value is None
        if value is None:
            return
        # Validate value ndim if possible
        value_ndim = A.ndim(value)
        if value_ndim is not None:
            value_shape = A.shape(value)
            assert value_ndim == 0, self._stamp_string("Expected loss scalar to be a scalar (0-D tensor), "
                                                       "got a {}-D tensor of shape {} instead.".
                                                       format(value_ndim, value_shape))
        self._loss_scalar = value

    def _get_loss_scalar(self):
        """Aggregate loss to a scalar."""
        return A.reduce_(self.loss_vector, mode=self.aggregation_method, name='loss_aggregation')

    def __call__(self, prediction, label):
        """Get the loss for all instances in a batch individually given the prediction and the label tensor."""
        return self.method(prediction, label)

    @application
    def apply(self, model):
        """Get the loss given a model and write as model attribute."""
        self.model = model
        py2.append_to_attribute(model, 'loss', self)

    @staticmethod
    def apply_weights(tensor, weights):
        return A.multiply(weights, tensor)

    def unbind_model(self):
        super(Loss, self).unbind_model()
        self.reset('loss_vector', 'loss_scalar')

    def attach_to_model_without_binding(self, model):
        py2.append_to_attribute(model, 'loss', self)

    def reset(self, *reset_what):
        _name_attribute_map = {'y': self._y,
                               'yt': self._yt,
                               'weights': self._weights,
                               'loss_vector': self._loss_vector,
                               'loss_scalar': self._loss_scalar}
        self._reset_attributes(reset_what, _name_attribute_map)


class Regularizer(ModelApp):
    """Abstract class for regularizer."""

    _ALLOWED_KWARGS = {'method', 'parameters', 'penalty_scalars',
                       'aggregation_method', 'coefficient', 'regularization_scalar'}

    def __init__(self, model=None, **kwargs):
        # Containers for properties
        self._model = None
        self._method = None
        self._parameters = None
        self._penalty_scalars = None
        self._aggregation_method = None
        self._coefficients = None
        self._regularization_scalar = None

        self.model = model

        # Validate kwargs
        self._validate_kwargs(**kwargs)

        self.parameters = kwargs.get('parameters')
        self.method = kwargs.get('method')
        self.penalty_scalars = kwargs.get('penalty_scalars')
        self.coefficients = kwargs.get('coefficients')
        self.aggregation_method = kwargs.get('aggregation_method', default='mean')
        self.regularization_scalar = kwargs.get('regularization_scalar')


    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self.reset('penalty_scalars', 'regularization_scalar')

    @property
    def parameters(self):
        if self.model is not None:
            # Fetch parameters from model if one is bound
            return self.model.parameters
        elif self._parameters is not None:
            return self._parameters
        else:
            raise RuntimeError(self._stamp_string("Parameters have not been defined yet. "
                                                  "To define `parameters`, consider assigning "
                                                  "to Regularizer.parameters, or provide a model."))

    @parameters.setter
    def parameters(self, value):
        # Ignore assignment if value is None
        if value is None:
            return
        # Parameter assignment is not possible if a model is bound
        if self.model is not None:
            warn(self._stamp_string('Setting `parameters` has no effect if `model` is bound. '
                                    'Consider unbinding the model first with `unbind_model` method.'),
                 RuntimeWarning)
        else:
            # Convert from parameter collection if required
            if hasattr(value, 'as_list'):
                value = value.as_list()
            # Make sure value is a list
            value = py.obj2list(value)
            self._parameters = value
            # Penality scalars need to be recomputed
            self.reset('penalty_scalars', 'regularization_scalar')

    @property
    def aggregation_method(self):
        return self._aggregation_method

    @aggregation_method.setter
    def aggregation_method(self, value):
        assert value in {'mean', 'sum'}, \
            self._stamp_string("Aggregation method must be "
                               "in ['mean', 'sum']. Got {} instead.".
                               format(value))
        self._aggregation_method = value
        self.reset('regularization_scalar')

    @property
    def method(self):
        if self._method is not None:
            return self._method
        else:
            raise RuntimeError(self._stamp_string("Regularizer `method` is yet to be set."))

    @method.setter
    def method(self, value):
        if value is None:
            return
        self._set_method(value)
        self.reset('penalty_scalars', 'regularization_scalar')

    def _set_method(self, method):
        # TODO Parse from string if required
        pass

    @property
    def coefficients(self):
        if self._coefficients is not None:
            return self._coefficients
        else:
            raise RuntimeError(self._stamp_string("Regularizer `coefficients` are yet to be set."))

    @coefficients.setter
    def coefficients(self, value):
        if value is None:
            return
        self._coefficients = value
        self.reset('regularization_scalar')

    @property
    def penalty_scalars(self):
        if self._penalty_scalars is not None:
            return self._penalty_scalars
        else:
            self._penalty_scalars = self._get_penalty_scalars()
            return self._penalty_scalars

    @penalty_scalars.setter
    def penalty_scalars(self, value):
        if value is None:
            return
        # Convert to list and validate length
        value = list(value)
        expected_len = len(self.parameters)
        if len(value) != expected_len:
            raise RuntimeError(self._stamp_string("The given `penalty_scalars` is of "
                                                  "length {}, but there are {} parameters.".
                                                  format(len(value), expected_len)))
        self._penalty_scalars = value
        # Regularization scalar needs to be recomputed, clear cache
        self.reset('regularization_scalar')

    def _get_penalty_scalars(self):
        return [self.method(parameter) for parameter in self.parameters]

    @property
    def regularization_scalar(self):
        if self._regularization_scalar is not None:
            return self._regularization_scalar
        else:
            self._regularization_scalar = self._get_regularization_scalar()
            return self._regularization_scalar

    @regularization_scalar.setter
    def regularization_scalar(self, value):
        # Do nothing if value is None
        if value is None:
            return
        # Validate value ndim if possible
        value_ndim = A.ndim(value)
        if value_ndim is not None:
            value_shape = A.shape(value)
            assert value_ndim == 0, self._stamp_string("Expected `regularization_scalar` to be a "
                                                       "scalar (0-D tensor), got a {}-D tensor of "
                                                       "shape {} instead.".
                                                       format(value_ndim, value_shape))
        self._regularization_scalar = value

    def _get_regularization_scalar(self):
        # Get penalty scalars
        penalty_scalars = self.penalty_scalars
        # Get and broadcast coefficients
        coefficients = self.coefficients
        if isinstance(coefficients, list):
            assert len(coefficients) == len(penalty_scalars) or len(coefficients) == 1, \
                self._stamp_string("Regularization coefficients must either be a scalar or a "
                                   "list of scalars of length {}. Got a list of length {} instead.".
                                   format(len(penalty_scalars), len(coefficients)))
        coefficients = py.broadcast(coefficients, len(self._penalty_scalars))
        weighted_penalty_scalars = [A.multiply(coefficient, penalty_scalar)
                                    for coefficient, penalty_scalar in zip(coefficients, penalty_scalars)]
        # Aggregate
        regularization_scalar = {'sum': A.add_n, 'mean': A.mean_n}[self.aggregation_method](weighted_penalty_scalars)
        return regularization_scalar

    @application
    def apply(self, model):
        """Get the loss given a model and write as model attribute."""
        self.model = model
        py2.append_to_attribute(model, 'regularization', self)

    def attach_to_model_without_binding(self, model):
        py2.append_to_attribute(model, 'regularizer', self)

    def reset(self, *reset_what):
        _name_attribute_map = {'parameters': self._parameters,
                               'coefficients': self._coefficients,
                               'penalty_scalars': self._penalty_scalars,
                               'regularization_scalar': self._regularization_scalar}
        self._reset_attributes(reset_what, _name_attribute_map)


class Objective(ModelApp):
    """Abstract class for the training objective. In general, objective = loss + regularization."""
    def __init__(self, model=None, **kwargs):
        # Property containers
        self._model = None
        self._losses = None
        self._regularizers = None
        self._objective_scalar = None
        self._parameters = None
        self._gradients = None

        self.model = model

    @application
    def apply(self, model):
        return model

    @property
    def losses(self):
        return NotImplemented

    @losses.setter
    def losses(self, value):
        raise NotImplementedError

    def add_loss(self, loss):
        pass

    @property
    def regularizers(self):
        return NotImplemented

    @regularizers.setter
    def regularizers(self, value):
        raise NotImplementedError

    def add_regularizer(self, regularizer):
        pass

    @property
    def objective_scalar(self):
        return NotImplemented

    @objective_scalar.setter
    def objective_scalar(self, value):
        raise NotImplementedError

    @property
    def parameters(self):
        return NotImplemented

    @parameters.setter
    def parameters(self, value):
        raise NotImplementedError

    @property
    def gradients(self):
        return NotImplemented

    @gradients.setter
    def gradients(self, value):
        raise NotImplementedError

    def reset(self, *reset_what):
        raise NotImplementedError


def apply(this, on):
    if hasattr(this, 'apply') and callable(this.apply):
        this.apply(on)
    else:
        raise NotImplementedError("Given `this` object (of class {}) "
                                  "doesn't have a callable `apply` method.".format(this.__class__.__name__))
