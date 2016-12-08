from .core import Trainer


class AsyncSynthGradTrainer(Trainer):
    # TODO: Asyncronously training the model with synthetic gradients, a la [1608.05343].
    # TODO: Investigate existing implementations:
    #           https://github.com/rarilurelo/tensorflow-synthetic_gradient
    #           https://github.com/andrewliao11/DNI-tensorflow
    pass
