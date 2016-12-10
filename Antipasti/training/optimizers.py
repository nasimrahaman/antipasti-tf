from .core import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer. Original Paper: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, **kwargs):
        super(Adam, self).__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

    # TODO Continue
