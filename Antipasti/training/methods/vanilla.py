from Antipasti.training.core import Trainer


class SupervisedTrainer(Trainer):
    """Trainer for training an Antipasti model with supervised methods."""
    # TODO
    def __init__(self):
        pass

    @property
    def objective(self):
        return None

    @objective.setter
    def objective(self, value):
        pass

    @property
    def optimizer(self):
        return None

    @optimizer.setter
    def optimizer(self, value):
        pass

    def fit(self):
        pass

    pass


class AsyncTrainer(SupervisedTrainer):
    """Trainer for training an Antipasti model asyncronously on multiple GPUs"""
    # TODO
    pass
