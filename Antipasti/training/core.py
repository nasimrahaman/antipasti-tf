__author__ = "Nasim Rahaman"

from ..utilities import utils
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

    _ALLOWED_KWARGS = {'aggregation_method'}

    def __init__(self, model=None, **kwargs):
        self.model = model

        # Read from kwargs
        self.aggregation_method = kwargs.get('aggregation_method', default='mean')

        # Property containers
        self._weights = None
        self._aggregation_method = None
        self._method = None
        self._loss_vector = None
        self._loss_scalar = None
        self._y = None
        self._yt = None

    @property
    def weights(self):
        """Weights for weighing the objective."""
        return self._weights

    @weights.setter
    def weights(self, value):
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

    @property
    def method(self):
        """Method. Should be a callable."""
        return self._method

    @method.setter
    def method(self, value):
        self._set_method(value)

    def _set_method(self, method, is_keras_objective=False):
        pass

    @property
    def y(self):
        # Check if y was defined
        if self._y is None:
            raise RuntimeError(self._stamp_string("`y` (prediction) is yet to be defined."))
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def yt(self):
        # Check if y was defined
        if self._yt is None:
            raise RuntimeError(self._stamp_string("`yt` (target) is yet to be defined."))
        return self._yt

    @yt.setter
    def yt(self, value):
        self._yt = value

    def assert_y_and_yt_shapes_are_compatible(self):
        y_shape = A.shape(self.y)
        yt_shape = A.shape(self.yt)
        assert utils.compare_shapes(y_shape, yt_shape), \
            self._stamp_string("Shape of prediction `y` (= {}) is not compatible "
                               "with that of target `yt` (= {}).".format(y_shape, yt_shape))

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
        # Validate value ndim if possible
        value_ndim = A.ndim(value)
        if value_ndim is not None:
            value_shape = A.shape(value)
            assert value_ndim == 1, self._stamp_string("Expected loss vector to be a vector (1-D tensor), "
                                                       "got a {}-D tensor of shape {} instead.".
                                                       format(value_ndim, value_shape))
        self._loss_vector = value

    def _get_loss_vector(self):
        """Get loss as a vector, such that its length equals the number of batches."""
        return self.method(self.y, self.yt)

    @property
    def loss_scalar(self):
        # Check if loss scalar is already cached
        if self._loss_scalar is not None:
            return self._loss_scalar
        else:
            self._loss_scalar = self._get_loss_scalar()

    @loss_scalar.setter
    def loss_scalar(self, value):
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
        # TODO
        return model


class Regularizer(ModelApp):
    """Abstract class for regularizer."""
    def __init__(self, model=None, **kwargs):
        self.model = model

    @application
    def apply(self, model):
        """Get the loss given a model and write as model attribute."""
        return model


class Objective(ModelApp):
    """Abstract class for the training objective. In general, objective = loss + regularization."""
    def __init__(self, model=None, **kwargs):
        self.model = model

    @application
    def apply(self, model):
        return model


def apply(this, on):
    if hasattr(this, 'apply') and callable(this.apply):
        this.apply(on)
    else:
        raise NotImplementedError("Given `this` object (of class {}) "
                                  "doesn't have a callable `apply` method.".format(this.__class__.__name__))
