__author__ = "Nasim Rahaman"


class Trainer(object):
    """Abstract class for an Antipasti Trainer."""
    # TODO
    pass


class TrainingConfiguration(object):
    """Configuration class for an Antipasti Trainer."""
    # TODO
    pass


class Optimizer(object):
    """Abstract class for an optimizer."""
    def __init__(self, **kwargs):
        pass

    def apply(self, model):
        """Get the training op and write as model attribute."""
        return model


class Loss(object):
    """Abstract class for a loss function (not neccesarily an objective)."""
    def __init__(self, **kwargs):
        self._weights = None
        self._method = None
        pass

    @property
    def weights(self):
        """Weights for weighing the objective."""
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def method(self):
        """Method. Should be a callable."""
        return self._method

    @method.setter
    def method(self, value):
        self.set_method(value)

    def set_method(self, method, is_keras_objective=False):
        pass

    def __call__(self, prediction, label):
        """Get the loss for all instances in a batch individually given the prediction and the label tensor."""
        return self.method(prediction=prediction, label=label)

    def get_scalar_loss(self):
        """Get aggregated the scalar."""
        return NotImplemented

    def apply(self, model):
        """Get the loss given a model and write as model attribute."""
        return model


class Regularizer(object):
    """Abstract class for regularizer."""
    def __init__(self, **kwargs):
        pass

    def apply(self, model):
        """Get the loss given a model and write as model attribute."""
        return model


class Objective(object):
    """Abstract class for the training objective. In general, objective = loss + regularization."""
    def __init__(self, **kwargs):
        pass

    def apply(self, model):
        return model


def apply(this, on):
    if hasattr(this, 'apply') and callable(this.apply):
        this.apply(on)
    else:
        raise NotImplementedError("Given `this` object (of class {}) "
                                  "doesn't have a callable `apply` method.".format(this.__class__.__name__))
