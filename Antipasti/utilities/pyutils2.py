"""Python Utilities. Functions in this module are not supposed to invoke the backend."""

import sys
import random
import string
import threading
import time
from datetime import datetime
from collections import OrderedDict
from Queue import Queue, Empty

from ..legacy import pykit as py

import numpy as np


# ---------------- FUNCTION-TOOLS ----------------


def vectorize_function(_string_stamper=None):
    """
    Decorator for vectorizing a function with proper broadcasting. Exercise extreme caution when using with
    functions that take lists as inputs.
    """
    # TODO Write moar doc

    # Default string stamper
    if _string_stamper is None:
        _string_stamper = lambda s: s

    def _vectorize_function(function):

        def _function(*args, **kwargs):
            # The first task is to get the vector length.
            vector_length = max([py.smartlen(arg) for arg in list(args) + list(kwargs.values())])

            # Make sure the given lists are consistent (i.e. smartlen either 1 or vector_length)
            assert all([py.smartlen(arg) == 1 or py.smartlen(arg) == vector_length
                        for arg in list(args) + list(kwargs.values())]), _string_stamper("Cannot broadcast arguments "
                                                                                         "/ vectorize function.")

            # Broadcast arguments
            broadcasted_args = [arg if py.smartlen(arg) == vector_length else py.broadcast(arg, vector_length)
                                for arg in args]

            # Broadcast keyword arguments <unreadable python-fu>
            broadcasted_kwargs = [[{key: value} for value in
                                   (kwargs[key] if py.smartlen(kwargs[key]) == vector_length else
                                    py.obj2list(kwargs[key]) * vector_length)]
                                  for key in kwargs.keys()]
            # </unreadable python-fu>

            # Output list
            outputs = []
            for arg, kwarg in zip(zip(*broadcasted_args), zip(*broadcasted_kwargs)):
                # kwarg is now a list of dictionaries. Put all these dicts to another, bigger dict
                big_kw_dict = dict([item for kw_dict in kwarg for item in kw_dict.items()])
                outputs.append(function(*arg, **big_kw_dict))

            return outputs

        return _function

    return _vectorize_function


# ---------------- COLLECTION-MANAGEMENT UTILITIES ----------------


def add_to_antipasti_collection(objects, **key_value_dict):
    """Populate objects' internal antipasti collection with (key, value) pairs from `key_value_dict`."""
    for object_ in py.obj2list(objects, ndarray2list=False):
        # Check if object has a collection dict already; if it doesn't, give it one
        if not hasattr(object_, '_antipasti_collection'):
            setattr(object_, '_antipasti_collection', {})
        # Update collection with key_value_dict (getattr to make the pycharm linter shut the fuck up)
        getattr(object_, '_antipasti_collection').update(key_value_dict)


def get_from_antipasti_collection(object_, key, default=None):
    """
    Get value for a given key in `object_`'s internal antipasti collection,
    and return `default` if key is not found.
    """
    if not hasattr(object_, '_antipasti_collection'):
        return default
    else:
        return getattr(object_, '_antipasti_collection').get(key, default)


def get_antipasti_collection(object_):
    """Gets the entire Antipasti collection dict of an object."""
    if not hasattr(object_, '_antipasti_collection'):
        setattr(object_, '_antipasti_collection', {})
    return getattr(object_, '_antipasti_collection')


def copy_antipasti_collection(from_, to_):
    """
    Copies Antipasti collection of object `from_` to object `to_`. Conflicting keys in `to_`
    will be overwritten.
    """
    collection_to_copy = get_antipasti_collection(from_)
    add_to_antipasti_collection(to_, **collection_to_copy)


def is_in_antipasti_collection(object_, key):
    """Checks whether a given key is in Antipasti collection of a given object."""
    return hasattr(object_, '_antipasti_collection') and key in getattr(object_, '_antipasti_collection').keys()


def is_antipasti_trainable(parameter):
    """Function to check if (Antipasti thinks) a parameter is trainable."""
    return get_from_antipasti_collection(parameter, 'trainable', default=True)


def filter_antipasti_trainable(parameters):
    """Function to weed out untrainable parameters given a list of `parameters`."""
    return [parameter for parameter in parameters if is_antipasti_trainable(parameter)]


def make_antipasti_trainable(parameters):
    """Make a parameter trainable with Antipasti."""
    # Get parameters as a list if passed as a parameter collection
    if hasattr(parameters, 'as_list'):
        parameters = parameters.as_list()
    add_to_antipasti_collection(parameters, trainable=True)


def make_antipasti_untrainable(parameters):
    """Make a parameter untrainable with Antipasti."""
    # Get parameters as a list if passed as a parameter collection
    if hasattr(parameters, 'as_list'):
        parameters = parameters.as_list()
    add_to_antipasti_collection(parameters, trainable=False)


def is_antipasti_regularizable(parameter):
    """Function to check if (Antipasti thinks) a parameter is regularizable."""
    return get_from_antipasti_collection(parameter, 'regularizable', default=True)


def filter_antipasti_regularizable(parameters):
    """Function to weed out unregularizable parameters given a list of `parameters`."""
    return [parameter for parameter in parameters if is_antipasti_regularizable(parameter)]


def make_antipasti_regularizable(parameters):
    """Make a parameter regularizable with Antipasti."""
    # Get parameters as a list if passed as a parameter collection
    if hasattr(parameters, 'as_list'):
        parameters = parameters.as_list()
    add_to_antipasti_collection(parameters, regularizable=True)


def make_antipasti_unregularizable(parameters):
    """Make a parameter regularizable with Antipasti."""
    # Get parameters as a list if passed as a parameter collection
    if hasattr(parameters, 'as_list'):
        parameters = parameters.as_list()
    add_to_antipasti_collection(parameters, regularizable=False)


# ---------------- PARAMETER-TAGS ----------------


def is_parameter_tag(tag):
    """
    Check if a tag (str) is a parameter tag. Parameter tags look like e.g.: '[LayerID:conv1][W]' for a layer named
    'conv1' and parameter named 'W'.
    """
    return isinstance(tag, str) and tag.startswith("[LayerID:") and tag.endswith("]") and tag.find('][') != -1


def split_parameter_tag(tag, check=False):
    """
    Splits a parameter tag to LayerID and parameter name.
    Example:
        split_parameter_tag('[LayerID:conv1][W]') -> ('conv1', 'W')
    """
    if check:
        assert is_parameter_tag(tag), "The tag to be split '{}' is not a valid parameter tag.".format(tag)
    # First, strip the exterior square brackets
    layer_id_tag, parameter_name = tag.strip('[]').split('][')
    # Get layer ID from tag
    layer_id = layer_id_tag.replace('LayerID:', '')
    # Done
    return layer_id, parameter_name


def get_parameter_tag(layer_id, parameter_name):
    """Gets parameter tag given a layer_id and a parameter name."""
    return "[LayerID:{}][{}]".format(layer_id, parameter_name)


def autoname_layer_or_model(layer=None, given_name=None, force_postfix=False,
                            _string_stamper=lambda s: s):
    # Check if the function has a library of used names. Make one if it doesn't.
    if not hasattr(autoname_layer_or_model, 'used_names'):
        autoname_layer_or_model.used_names = {}
    # Validate
    if hasattr(layer, 'name_is_user_defined'):
        assert not layer.name_is_user_defined, \
            _string_stamper("Autonaming failed: layer or model name was found to"
                            " be user defined as '{}'.".format(layer.name))
    assert given_name is None or isinstance(given_name, str), \
        _string_stamper("Given name must be a string.")
    assert not (layer is None and given_name is None), \
        _string_stamper("Either a `layer` or a `given_name` must be "
                        "provided for autonaming to work.")
    # Get class prefix
    name_prefix = layer.__class__.__name__.lower() if given_name is None else given_name
    # Get object id as the id of the layer if it's given
    if layer is not None:
        object_id = str(id(layer))
    else:
        # python id's are pretty big, so chances of collision are slim #yolo
        object_id = autoname_layer_or_model.used_names.get(name_prefix, 0)
        autoname_layer_or_model.used_names.update({name_prefix: object_id + 1})
    # Make name and return. If object_id is 0, there's no need to postfix
    if object_id != 0 or force_postfix:
        return "{}_{}".format(name_prefix, object_id)
    else:
        return name_prefix


# ---------------- INSTANCE-MANIPULATION-TOOLS ----------------


def append_to_attribute(object_, attribute_name, attribute_object, delist=True,
                        prevent_duplicates=False):
    """
    If a given object has an attribute named `attribute_name`, which happens to be a list,
    this function appends `attribute_object` to it. If the attribute is not present,
    this function creates one. If it's present but not a list, it's made one. The `delist`
    argument specifies if the first object appended is a list or simply the given object
    `attribute_object`. Finally, setting `prevent_duplicates` to true will result in
    `attribute_object` not being appended to the list if it's in there already.
    """
    _LISTLIKE = (list,)

    if hasattr(object_, attribute_name):
        # Object has the attribute
        if isinstance(getattr(object_, attribute_name), _LISTLIKE):
            # Attribute is a list, append to it directly if not in already
            (py.appendunique if prevent_duplicates else list.append)\
                (list(getattr(object_, attribute_name)), attribute_object)
        else:
            # Attribute is not listlike, it must be made one.
            attribute_list = py.obj2list(getattr(object_, attribute_name))
            # Append to the new attribute_list
            (py.appendunique if prevent_duplicates else list.append)\
                (attribute_list, attribute_object)
            # Write attribute
            setattr(object_,
                    attribute_name,
                    (py.delist(attribute_list) if delist else attribute_list))
    else:
        # Object does not have the attribute - set it.
        setattr(object_, attribute_name, (attribute_object if delist else [attribute_object]))


# ---------------- ADVANCED-DSTRUCTS ----------------


class DictList(OrderedDict):
    """
    This class brings some of the list goodies to OrderedDict (including number indexing), with the caveat that
    keys are only allowed to be strings.
    """

    def __init__(self, item_list, **kwds):
        # Try to make item_list compatible without looking for key conflicts
        item_list = self._make_compatible(item_list, find_key_conflicts=False)
        # Init superclass
        super(DictList, self).__init__(item_list, **kwds)
        # Raise exception if non-string found in keys
        if not all([isinstance(key, str) for key in self.keys()]):
            raise TypeError("Keys in a DictList must be string.")

    def __setitem__(self, key, value, dict_setitem=dict.__setitem__):
        # This method is overridden to intercept keys and check whether they're strings
        if not isinstance(key, str):
            raise TypeError("Keys in a DictList must be strings.")
        super(DictList, self).__setitem__(key, value, dict_setitem=dict_setitem)

    def __getitem__(self, item):
        # This is where things get interesting. This function is overridden to enable number indexing.
        if not isinstance(item, (str, int, slice)):
            raise TypeError("DictList indices must be slices, integers "
                            "or strings, not {}.".format(item.__class__.__name__))
        # Case one: item is a string
        if isinstance(item, str):
            # Fall back to the superclass' getitem
            return super(DictList, self).__getitem__(item)
        else:
            # item is an integer. Fetch from list and return
            return self.values()[item]

    def _is_compatible(self, obj, find_key_conflicts=True):
        """Checks if a given object is convertable to OrderedDict."""
        # Check types
        if isinstance(obj, (OrderedDict, DictList, dict)):
            code = 1
        elif isinstance(obj, list):
            code = 2 if all([py.smartlen(elem) == 2 for elem in obj]) else 3
        else:
            code = 0

        # Check for key conflicts
        if find_key_conflicts and (code == 1 or code == 2):
            if not set(self.keys()) - set(OrderedDict(obj).keys()):
                # Key conflict found, obj not compatible
                code = 0
        # Done.
        return code

    def _make_compatible(self, obj, find_key_conflicts=True):
        # Get compatibility code.
        code = self._is_compatible(obj, find_key_conflicts=find_key_conflicts)

        # Convert code 3
        if code == 3:
            compatible_obj = []
            for elem in obj:
                taken_keys = zip(*compatible_obj)[0] if compatible_obj else None
                generated_id = self._generate_id(taken_keys=taken_keys)
                compatible_obj.append((generated_id, elem))
            obj = compatible_obj
        elif code == 1 or code == 2:
            # Object is compatible already, nothing to do here.
            pass
        else:
            raise ValueError("Object could not be made compatible with DictList.")

        return obj

    def append(self, x):
        # This is custom behaviour.
        # This method 'appends' x to the dict, but without a given key.
        # This is done by setting str(id(x)) as the dict key.
        self.update({self._generate_id(taken_keys=self.keys()): x})

    def extend(self, t):
        # Try to make t is compatible
        t = self._make_compatible(t)
        # Convert t to list, and piggy back on the superclass' update method
        self.update(list(t))

    def __add__(self, other):
        # Enable list concatenation with +
        # Try to make other compatible
        self._make_compatible(other)
        # Use OrderedDict constructor
        return DictList(self.items() + list(other))

    @staticmethod
    def _generate_id(taken_keys=None):
        _SIZE = 10
        taken_keys = [] if taken_keys is None else taken_keys

        while True:
            generated_id = ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits)
                                   for _ in range(_SIZE))
            # If generated_id is not taken, break and return, otherwise, retry.
            if generated_id not in taken_keys:
                return generated_id
            else:
                continue


class ParameterCollection(DictList):
    """Class to collect parameters of a layer."""
    def __init__(self, item_list, **kwds):
        # Initialize superclass
        super(ParameterCollection, self).__init__(item_list, **kwds)
        # Validate contents of the built
        self._validate_items()

    def __getitem__(self, item):
        # Check if item is a string to start with

        if isinstance(item, str):
            # So far so good. Now check whether it's a parameter tag
            if self._is_parameter_tag(item):
                return super(ParameterCollection, self).__getitem__(item)
            else:
                # FIXME This is about as inefficient as it gets. I know, a few seconds do not matter
                #       if you're training a network, but there has to be a better way
                #       (e.g. caching names and layer_id's).
                # Check if it's a parameter name.
                names_found = self.find(parameter_name=item)
                # Check if it's a layer id
                layers_found = self.find(layer_id=item)
                # Check if item is both a layerID and parameter name
                if bool(names_found) ^ bool(layers_found):
                    return py.delist(names_found) if names_found else py.delist(layers_found)
                else:
                    # item is both a layerID and parameter name
                    #
                    raise KeyError("Item(s) {} is(are) both LayerID(s) and parameter name(s). "
                                   "Resolve conflict by using parameter tags.".format(names_found))
        else:
            # Let the superclass handle this mess
            return super(ParameterCollection, self).__getitem__(item)

    def find(self, layer_id=None, parameter_name=None):
        # Enforce early stopping if both layer_id and parameter_name is given
        stop_when_found = layer_id is not None and parameter_name is not None
        # Instantiate a list to put search results in
        found = []
        # Search
        for item_key, item_value in self.items():
            current_layer_id, current_parameter_name = self._split_parameter_tag(item_key, check=True)
            # Check if there's a match
            layer_id_match = True if layer_id is None else layer_id == current_layer_id
            parameter_name_match = True if parameter_name is None else parameter_name == current_parameter_name
            # Append to found if there is a match, keep looking otherwise
            if layer_id_match and parameter_name_match:
                found.append(item_value)
                if stop_when_found:
                    break
            else:
                continue
        # Done
        return found

    def __setitem__(self, key, value):
        # Check if key is a parameter tag
        if not self._is_parameter_tag(key):
            raise ValueError("Key {} is not a parameter tag.".format(key))
        super(ParameterCollection, self).__setitem__(key, value)

    def set(self, layer_id, parameter_name, value):
        self.__setitem__(self._get_parameter_tag(layer_id, parameter_name), value)

    def as_list(self):
        return self.values()

    def _validate_items(self, items=None):
        # Use items in the dict if items is not given
        items = self.items() if items is None else items
        for item_key, item_value in items:
            if not self._is_parameter_tag(item_key):
                raise ValueError("Key {} is not a valid parameter tag.".format(item_key))

    _is_parameter_tag = is_parameter_tag
    _split_parameter_tag = split_parameter_tag
    _get_parameter_tag = get_parameter_tag


# ---------------- DEBUG-UTILITIES ----------------


class DebugLogger(object):
    def __init__(self, object_name, output_stream=None, activate=True):
        # Private variables
        self._output_stream = None
        self._object_name = None
        self._is_active = activate
        self._write_lock = threading.RLock()
        # Assignments
        self.object_name = object_name
        self.output_stream = output_stream

    @property
    def output_stream(self):
        if self._output_stream is None:
            self._output_stream = sys.stdout
        return self._output_stream

    @output_stream.setter
    def output_stream(self, value):
        if value is not None:
            assert hasattr(value, 'write') and callable(value.write), \
                "`output_stream` must have a callable `write` attribute."
            self._output_stream = value

    @property
    def object_name(self):
        return self._object_name

    @object_name.setter
    def object_name(self, value):
        assert isinstance(value, str), \
            "`object_name` must be a string, got {} instead.".format(value.__class__.__name__)
        self._object_name = value

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False

    def log(self, message, method_name=None, thread_num=None):
        if self._is_active:
            # Get thread name from python if none provided
            thread_num = threading.current_thread().ident if thread_num is None else thread_num
            log_message = "[{}] [{}{}{}] {}\n".\
                format(str(datetime.now()),
                       self.object_name,
                       ".{}".format(method_name) if method_name is not None else '',
                       "::thread_{}".format(thread_num) if thread_num is not None else '',
                       message)
            with self._write_lock:
                self.output_stream.write(log_message)

    def get_logger_for(self, method_name=None, thread_num=None):
        return _MethodLogger(self, method_name=method_name, thread_num=thread_num)

    def clean_up(self):
        """Close open file streams."""
        self.output_stream.close()


class _MethodLogger(object):
    def __init__(self, debug_logger, method_name=None, thread_num=None):
        """
        :type debug_logger: DebugLogger
        :param debug_logger: Debug Logger to go with this _MethodLogger.

        :type method_name: str
        :param method_name: Name of the method this logger is to be called from.

        :type thread_num: str
        :param thread_num: Thread this logger is to be called from.
        """
        # Private
        self._analysis_lambdas = {'type': lambda obj: type(obj)}
        self._method_logger_is_active = True
        # Not private
        self.debug_logger = debug_logger
        self.method_name = method_name
        self.thread_num = thread_num

    def activate(self):
        self._method_logger_is_active = True

    def deactivate(self):
        self._method_logger_is_active = False

    def add_analysis_lambdas(self, **analysis_lambdas):
        assert all([callable(lamb) for lamb in analysis_lambdas.values()]), \
            "`analysis_lambdas` must be a dictionary with callable values."
        self._analysis_lambdas.update(analysis_lambdas)

    def remove_analysis_lambdas(self, *analysis_lambdas):
        for analysis_lambda in analysis_lambdas:
            self._analysis_lambdas.pop(analysis_lambda)

    def __call__(self, message):
        if self._method_logger_is_active:
            self.debug_logger.log(message,
                                  method_name=self.method_name,
                                  thread_num=self.thread_num)

    def analyze(self, object_, object_name=None, **extra_analysis_lambdas):
        # No analysis if method_logger is not active
        if not self._method_logger_is_active:
            return
        analysis_string = "[Analysis: {}] ".format(object_name if object_name is not None else '')
        all_analysis_lambdas = dict(self._analysis_lambdas.items() + extra_analysis_lambdas.items())
        for analysis_name, analysis_lambda in all_analysis_lambdas.items():
            try:
                analysis_result = analysis_lambda(object_)
            except:
                analysis_result = "FAILED"
            analysis_string += "| {} :: {} |".format(analysis_name, analysis_result)
        self.__call__(analysis_string)

    @staticmethod
    def autofetch_analysis_lambdas(object_):
        _is_np_array = lambda obj: isinstance(obj, np.ndarray)

        # Get object type
        if isinstance(object_, (list, tuple)):
            # List or tuple
            if all(_is_np_array(elem) for elem in object_):
                # List or tuple of numpy arrays
                object_type = 'list[ndarray]'
            else:
                # Just a plain old list or tuple
                object_type = 'list'
        elif _is_np_array(object_):
            # Numpy array
            object_type = 'ndarray'
        else:
            object_type = None

        # Lambda library
        lambda_library = {'list[ndarray]': {'length': len,
                                            'shape_of_first_element': lambda x: x[0].shape,
                                            'shape_of_last_element': lambda x: x[-1].shape},

                          'list': {'length': len,
                                   'dtype_of_first_element': lambda x: type(x[0]),
                                   'dtype_of_last_element': lambda x: type(x[-1])},

                          'ndarray': {'shape': lambda x: x.shape,
                                      'max': lambda x: x.max(),
                                      'mean': lambda x: x.mean(),
                                      'min': lambda x: x.min()}}
        return lambda_library.get(object_type, {})


class MultiplexedFileStream(object):
    """Bundle multiple file-streams to one stream-like object."""
    def __init__(self, *streams):
        self.streams = list(streams)

    def write(self, p_str):
        for stream in self.streams:
            stream.write(p_str)

    def close(self):
        for stream in self.streams:
            stream.close()


class BufferedFunction(object):
    def __init__(self, target, num_threads=1, interrupt_event=None, latency=1):
        # Private
        self._inbound_queue = Queue()
        self._outbound_queue = Queue()
        self._agent_spec = []
        self._put_count = 0
        self._get_count = 0
        self._target_is_running = threading.Event()
        # Public
        self.target = target
        self.num_threads = num_threads
        self.interrupt_event = interrupt_event if interrupt_event is not None else threading.Event()
        self.latency = latency

    def _decorate_target(self, target):
        def agent(inbound_queue, outbound_queue):
            # Main thread loop
            while True:
                # Break if interrupt event is set
                if self.interrupt_event.is_set():
                    break

                # Try to fetch from queue
                try:
                    from_queue = inbound_queue.get(timeout=self.latency)
                except Empty:
                    # Move on if nothing in queue
                    continue

                if isinstance(from_queue, PoisonPill):
                    break
                try:
                    # Set target running event
                    self._target_is_running.set()
                    # Pass what's fetched to the target
                    from_target = target(from_queue)
                    # Clear target running event
                    self._target_is_running.clear()
                except Exception:
                    # Clear target running event
                    self._target_is_running.clear()
                    # Uh-oh
                    # Interrupt all other threads
                    self.interrupt_event.set()
                    raise
                # Put to outbound queue
                outbound_queue.put(from_target)
            return
        return agent

    def start(self):
        """Start all computation threads."""
        # Fetch agents
        for thread_num in range(self.num_threads):
            # Make agent by decorating target
            agent = self._decorate_target(self.target)
            agent_thread = threading.Thread(target=agent,
                                            args=(self._inbound_queue, self._outbound_queue))
            self._agent_spec.append({'thread_num': thread_num,
                                     'agent': agent,
                                     'thread': agent_thread})
            # Start agent thread
            agent_thread.start()

    def stop(self):
        """Interrupt all computation threads."""
        # Set interrupt event
        self.interrupt_event.set()
        # Join
        self.join()

    def join(self):
        """Join all computation threads."""
        for agent_spec in self._agent_spec:
            agent_spec['thread'].join()

    def is_alive(self):
        """Checks if threads are defined and alive."""
        if len(self._agent_spec) == 0:
            return False
        else:
            return all(agent_spec['thread'].is_alive() for agent_spec in self._agent_spec)

    def put(self, item):
        """Enqueues an item for execution."""
        self._put_count += 1
        self._inbound_queue.put(item)

    def size(self):
        return self._put_count - self._get_count

    def get(self, timeout=None):
        # Check if timeout is None. If it is, try getting permanently,
        # ocassionally checking for interrupts. If not, try getting only
        # once with the given timeout.
        if timeout is not None:
            retry_getting = False
        else:
            retry_getting = True
            timeout = self.latency
        # Check if there are enough items in the queue, or this function call could result
        # in a deadlock.
        if self._get_count >= self._put_count:
            self.interrupt_event.set()
            raise RuntimeError("Not enough items in the queue.")

        while True:
            if self.interrupt_event.is_set():
                break
            try:
                out = self._outbound_queue.get(timeout=timeout)
                self._get_count += 1
                return out
            except Empty:
                if retry_getting:
                    continue
                else:
                    return

    def done(self):
        """Kill threads when they're done."""
        # We'll need as many poison pills as agents
        for _ in range(self.num_threads):
            self.put(PoisonPill())

    def stop_when_done(self):
        """
        Stops all threads when they're finished, i.e. when there are no jobs enqueued in the
        inbound queue and no targets are running.
        """
        while True:
            if self._inbound_queue.qsize() == 0 and not self._target_is_running.is_set():
                self.interrupt_event.set()
                break
            else:
                time.sleep(self.latency)
        self.join()


class PoisonPill(object):
    pass
