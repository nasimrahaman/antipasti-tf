import time
import numpy as np
import Antipasti.io.runners as runners
import Antipasti.backend as A


def test_feeder_runner():
    # ---- BASIC TEST
    # Define a random generator to mimic mnist
    def random_generator_mnist(batch_size=10):
        yield np.random.uniform(size=(batch_size, 28, 28, 1)).astype('float32'), \
              np.random.uniform(size=(batch_size, 1, 1, 1)).astype('float32')

    # Build preprocessing function
    def preprocessor(inputs):
        time.sleep(5)
        return inputs

    # Build feeder runner with multiple threads. Observe that the batch_size of the generator need
    # not be that of the feeder_runner
    feeder_runner = runners.FeederRunner(feeder=random_generator_mnist(),
                                         preprocessor=preprocessor, dtypes=['float32', 'float32'],
                                         input_shape=[[None, 28, 28, 1], [None, 1, 1, 1]],
                                         num_threads=2, batch_size=5)
    images, labels = feeder_runner.dq()

    # We need to make a session right now (in this thread) because tensorflow would set a different
    # default session for every thread
    # sess = A.Session.session

    # Start generator threads
    feeder_runner.start_runner()

    # Dequeue
    np_images, np_labels = A.run([images, labels])

    # Stop generator threads
    feeder_runner.stop_runner()

    # Check shapes
    assert np_images.shape == (5, 28, 28, 1)
    assert np_labels.shape == (5, 1, 1, 1)
