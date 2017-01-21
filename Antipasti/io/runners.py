__author__ = "Nasim Rahaman"

import threading

from .. import backend as A
from ..utilities import utils
from ..utilities import pyutils2 as py2
from ..legacy import pykit as py


class FeederRunner(object):
    """
    Class to enqueue Tensorflow feeders from a datafeeder.
    Loosely inspired by https://indico.io/blog/tensorflow-data-input-part2-extensions/
    """
    def __init__(self, feeder, batch_size=1, preprocessor=None, dtypes=None, num_threads=1, num_epochs_per_thread=1,
                 queue_capacity=2000, min_num_elements_in_queue=0, coordinator=None,
                 dimensions=None, num_inputs=None, input_shape=None):
        # Property containers and internal variables
        self._dtypes = None
        self._preprocessor = None
        self._queue = None
        self._queue_size_op = None
        self._enq_op = None
        self._data_placeholders = None
        self._coordinator = None
        self._debug_logger = None

        self.feeder_lock = threading.Lock()

        # Assignments
        self.feeder = feeder
        self.queue_capacity = queue_capacity
        self.min_num_elements_in_queue = min_num_elements_in_queue
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.num_epochs_per_thread = num_epochs_per_thread

        # Get input shapes ('abuse' utils.get_input_shape)
        self.input_shape = utils.get_input_shape(dimensions=dimensions, num_inputs=num_inputs,
                                                 known_input_shape=input_shape,
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

    @property
    def thread_list(self):
        return list(self.coordinator._registered_threads)

    @property
    def queue_is_made(self):
        """Checks if the queue is defined."""
        return self._queue is not None

    def make_queue(self):
        """Finalize and make a queue."""
        self._queue = A.getfw().RandomShuffleQueue(shapes=[_input_shape[1:] for _input_shape in self.input_shapes],
                                                   dtypes=self.dtypes,
                                                   capacity=self.queue_capacity,
                                                   min_after_dequeue=self.min_num_elements_in_queue)

        # Make placeholders for data tensors on the cpu (data stuff goes in the CPU)
        self._data_placeholders = [A.placeholder(dtype=dtype, shape=shape, device='cpu')
                                   for shape, dtype in zip(self.input_shapes, self.dtypes)]

        # Make enqueue op
        self._enq_op = self._queue.enqueue_many(self._data_placeholders)

    def _make_queue_size_op(self):
        if not self.queue_is_made:
            raise RuntimeError("Queue is yet to be defined. Consider "
                               "calling the make_queue method first.")
        self._queue_size_op = self.queue.size()

    def get_queue_size(self, symbolic=False, session=None):
        """
        Gets size of the queue. If `symbolic` is set to True, a symbolic queue-size op is
        returned. Otherwise, the op is evaluated with `session` (or the default Antipasti session
        if none provided).
        """
        # Make size op
        if self._queue_size_op is None:
            self._make_queue_size_op()
        # Return op if symbolic
        if symbolic:
            return self._queue_size_op
        else:
            # Get default session from backend
            session = A.Session.session if session is None else session
            return session.run(self._queue_size_op)

    @property
    def queue_size(self):
        return self.get_queue_size(symbolic=False)

    def dq(self):
        """Get `Tensor`s resulting from dequeue'ing the queue."""
        dqd = self.queue.dequeue_many(self.batch_size)
        return dqd

    def nq(self, session=None, thread_num=None):
        """Read data from feeder, apply preprocessor and enqueue. This function is executed by a single thread."""
        # Get debug logger function
        log = self.debug_logger.get_logger_for('nq', thread_num)

        log("Getting session")
        # Get default session from backend
        session = A.Session.session if session is None else session

        try:
            # Iterate and add to feeder
            for epoch_num in range(self.num_epochs_per_thread):
                log("Starting epoch {} of {}".format(epoch_num, self.num_epochs_per_thread))
                # Check if we need to break out of the loop
                if self.coordinator.should_stop():
                    log("Got stop signal from coordinator, breaking from epoch-loop")
                    break
                # Data loop
                while True:
                    # Check if we need to break out of the loop
                    if self.coordinator.should_stop():
                        log("Got stop signal from coordinator, breaking from feeder-loop")
                        break

                    # Get data batch with lock
                    self.feeder_lock.acquire()
                    log("Acquired feeder lock")
                    try:
                        # (if the preproccessing is not done here, this shouldn't take long)
                        log("Trying to get batch from feeder")
                        data_batch = self.feeder.next()
                        log("Got batch from feeder")
                    except StopIteration:
                        log("StopIteration from feeder, breaking")
                        break
                    finally:
                        # Release lock
                        log("Releasing feeder lock")
                        self.feeder_lock.release()

                    log("Preprocessing")
                    # Preprocess data_batch
                    prepped_data_batch = self.preprocessor(data_batch)

                    # Validate data_batch
                    assert len(prepped_data_batch) == self.num_inputs, \
                        "Data batch as yielded by the feeder (after preprocessing) has {} tensors, " \
                        "but this FeederRunner instance expects {}.".format(len(prepped_data_batch),
                                                                            self.num_inputs)

                    # Get feed dict
                    feed_dict = {_placeholder: _data_tensor
                                 for _placeholder, _data_tensor in zip(self._data_placeholders, prepped_data_batch)}
                    log("Enqueuing")
                    # NQ
                    session.run(self._enq_op, feed_dict=feed_dict)
                    log("Enqueued")
                # Restart feeder for the next epoch (if possible, break otherwise)
                if hasattr(self.feeder, 'restart_generator'):
                    log("Restarting generator")
                    self.feeder.restart_generator()
                elif hasattr(self.feeder, 'restartgenerator'):
                    log("Restarting generator (legacy)")
                    # Legacy support
                    self.feeder.restartgenerator()
                else:
                    log("Breaking")
                    break

        except Exception as e:
            log("Exception raised, requesting coordinator to stop")
            log("The exception message follows: {}".format(e.message))
            self.coordinator.request_stop(e)

    def weave_threads(self, session=None):
        """Start datafeeder threads."""
        # Get default session from backend (if none provided)
        session = A.Session.session if session is None else session
        # Start threads
        for thread_num in range(self.num_threads):
            thread = threading.Thread(target=self.nq, args=(session, thread_num))
            thread.daemon = True
            thread.start()
            self.coordinator.register_thread(thread=thread)

    def start_runner(self, session=None):
        """Start queue runners and the feeder threads."""
        # Get default session from backend (if none provided)
        session = A.Session.session if session is None else session
        # Start tensorflow queue runners
        A.getfw().train.start_queue_runners(sess=session)
        # Start feeder threads
        self.weave_threads(session=session)

    def stop_runner(self):
        """Try to stop all running threads."""
        self.coordinator.request_stop()

    def join_runner(self):
        """Stop all threads and wait for them to finish."""
        self.coordinator.join()

    @property
    def debug_logger(self):
        """Get logger for debug messages."""
        if self._debug_logger is None:
            self._debug_logger = py2.DebugLogger(object_name='FeederRunner', activate=False)
        return self._debug_logger

    @debug_logger.setter
    def debug_logger(self, value):
        if value is not None:
            assert isinstance(value, py2.DebugLogger), \
                "`debug_logger` must be a `DebugLogger` object, " \
                "got {} instead.".format(value.__class__.__name__)
            self._debug_logger = value

