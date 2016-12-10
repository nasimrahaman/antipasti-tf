__author__ = "Nasim Rahaman"

from .. import backend as A
from ..utilities import utils
from ..legacy import pykit as py


class FeederRunner(object):
    """
    Class to enqueue Tensorflow feeders from a datafeeder.
    Loosely inspired by https://indico.io/blog/tensorflow-data-input-part2-extensions/
    """
    def __init__(self, feeder, batch_size=1, preprocessor=None, dtypes=None, num_threads=1, num_epochs_per_thread=1,
                 queue_capacity=2000, min_num_elements_in_queue=1000, coordinator=None,
                 dimensions=None, num_inputs=None, input_shape=None):
        # Property containers and internal variables
        self._dtypes = None
        self._preprocessor = None
        self._queue = None
        self._enq_op = None
        self._data_placeholders = None
        self._coordinator = None

        # Assignments
        self.feeder = feeder
        self.queue_capacity = queue_capacity
        self.min_num_elements_in_queue = min_num_elements_in_queue
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_epochs_per_thread = num_epochs_per_thread

        # Get input shapes ('abuse' utils.get_input_shape)
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=num_inputs,
                                                 known_input_shape=py.delistlistoflists(input_shape),
                                                 default_num_inputs=2, default_dimensions=2)

        # Set dtypes
        self.dtypes = dtypes

        # Set preprocessor
        self.preprocessor = preprocessor

        # Set coordinator
        self.coordinator = coordinator

    @property
    def input_shapes(self):
        return py.list2listoflists(self.input_shape)

    @property
    def dtypes(self):
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value):
        # Get datatypes
        if value is None:
            # Default to floatX if nothing provided
            self._dtypes = [A._FLOATX for _ in range(len(self.input_shapes))]
        else:
            assert py.smartlen(value) == 1 or py.smartlen(value) == len(self.input_shapes), \
                "dtypes must either be a list with 1 or `num_inputs` elements specifying the " \
                "dtypes of individual inputs, or a tensorflow datatype object or a string " \
                "specifying the dtypes of all inputs."
            # Broadcast to a list of the right length and convert to tensorflow datatype
            self._dtypes = [A.to_tf_dtype(dtype) for dtype in py.broadcast(value, self.num_inputs)]

    @property
    def preprocessor(self):
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        if value is None:
            value = lambda x: x
        else:
            assert callable(value), \
                "Preprocessor value must be set to a callable. " \
                "The class {} is not callable.".format(value.__class__.__name__)
        self._preprocessor = value

    @property
    def coordinator(self):
        if self._coordinator is None:
            self.coordinator = None
            return self._coordinator
        else:
            return self._coordinator

    @coordinator.setter
    def coordinator(self, value):
        if value is None:
            # Make coordinator
            self._coordinator = A.getfw().train.Coordinator()
        else:
            self._coordinator = value

    @property
    def num_inputs(self):
        return len(self.input_shapes)

    @property
    def queue(self):
        # Make queue if it's not made already
        if self._queue is None:
            return self.make_queue()
        else:
            return self._queue

    def make_queue(self):
        """Finalize and make a queue."""
        self._queue = A.getfw().RandomShuffleQueue(shapes=self.input_shapes,
                                                   dtypes=self.dtypes,
                                                   capacity=self.queue_capacity,
                                                   min_after_dequeue=self.min_num_elements_in_queue)

        # Make placeholders for data tensors on the cpu (data stuff goes in the CPU)
        self._data_placeholders = [A.placeholder(dtype=dtype, shape=shape, device='cpu')
                                   for shape, dtype in zip(self.input_shapes, self.dtypes)]

        # Make enqueue op
        self._enq_op = self._queue.enqueue_many(self._data_placeholders)

    def dq(self):
        """Get `Tensor`s resulting from dequeue'ing the queue."""
        dqd = self.queue.dequeue_many(self.batch_size)
        return dqd

    def nq(self, sess=None):
        """Read data from feeder, apply preprocessor and enqueue. This function is executed by a single thread."""
        # Get default session from backend
        sess = A.Session.session if sess is None else sess

        try:
            # Iterate and add to feeder
            for epoch_num in range(self.num_epochs_per_thread):
                # Check if we need to break out of the loop
                if self.coordinator.should_stop():
                    break
                # Data loop
                for data_batch in self.feeder:
                    # Check if we need to break out of the loop
                    if self.coordinator.should_stop():
                        break

                    # Validate data_batch
                    assert len(data_batch) == self.num_inputs, \
                        "Data batch as yielded by the feeder has {} tensors, " \
                        "but this FeederRunner instance expects {}.".format(len(data_batch), self.num_inputs)

                    # Preprocess data_batch
                    prepped_data_batch = self.preprocessor(data_batch)
                    # Get feed dict
                    feed_dict = {_placeholder: _data_tensor
                                 for _placeholder, _data_tensor in zip(self._data_placeholders, prepped_data_batch)}
                    # NQ
                    sess.run(self._enq_op, feed_dict=feed_dict)

                # Restart feeder for the next epoch
                self.feeder.restart_generator()

        except Exception as e:
            self.coordinator.request_stop(e)

    def weave(self):
        # TODO Make threads
        pass
