__author__ = "Nasim Rahaman"

from collections import OrderedDict
from contextlib2 import ExitStack

import random
import string

try:
    from . import pyutils as py
except ValueError:
    # Try to import as a standalone
    import pyutils as py


def forward_pass(forward_function):
    """
    Decorator for the feedforward method of `Layer`. The `feedforward` method must be able to handle input as a
    keyword argument; this decorator ensures that any given `forward_function` is always fed its input - in other
    words, `forward_function` can be allowed to handle its input as an argument (and not necessarily a keyword
    argument).
    """
    def _feedforward(cls, input=None):
        # Define behaviour for when input is None:
        if input is None:
            input = cls.x
        else:
            cls.x = input
        # Evaluate output
        output = forward_function(cls, input=input)
        # Assign output to y
        cls.y = output
        # Return output
        return output
    # Return decorated function
    return _feedforward


def shape_inference(shape_inference_function):
    """
    Decorator for the `infer_output_shape` method of `Layer`. The motivation for this decorator is the same as that
    of the `forward_pass` decorator.
    """

    def _infer_output_shape(cls, input_shape=None):
        if input_shape is None:
            input_shape = cls.input_shape
        return shape_inference_function(cls, input_shape=input_shape)

    return _infer_output_shape


def call_in_managers(context_managers=None):
    """
    Decorator factory that makes a decorator to call the decorated function within nested `context_managers`.

    :type context_managers: list
    :param context_managers: List of context managers to nest over. The first manager in list is entered first.
    """
    def _decorator(function):
        def decorated_function(*args, **kwargs):
            with ExitStack as stack:
                # Enter managers
                for manager in context_managers:
                    stack.enter_context(manager)
                # Evaluate function
                output = function(*args, **kwargs)
            return output
        return decorated_function
    return _decorator


# This function is best left folded.
def get_input_shape(dimensions=None, num_inputs=None, known_input_shape=None, num_features_in=None,
                    _string_stamper=None, default_dimensions=2, default_num_inputs=1):
    """Deduce input_shape (in NHWC (byxc) or NDHWC (bzyxc) format) from what's known."""
    if _string_stamper is None:
        _string_stamper = lambda s: s

    _dimension_to_tensor_dimension = {2: 4, 3: 5}
    _tensor_dimension_to_dimension = {4: 2, 5: 3}

    # If an input shape is given:
    if known_input_shape is None:
        # If dimensions is not given, default to the default value
        if dimensions is None:
            # Make sure default_dimensions is not set to None
            assert default_dimensions is not None, _string_stamper("`default_dimensions` must not be None for the "
                                                                   "input_shape parsing to work.")
            dimensions = default_dimensions

        # Check if num_inputs are given
        if num_inputs is None:
            # If not, try to get one from dimensions if possible
            if isinstance(dimensions, list):
                # If dimensions is a list, we're clear on the number of inputs
                num_inputs = len(dimensions)
                # Now, dimensions could be a list with None's. The list is basically there to convey num_inputs.
                # In this case, replace None's with the default_dimensions. But before proceeding, we need to be sure
                # that default_dimensions itself is a list.
                if not isinstance(default_dimensions, list):
                    # If it's not a list, use the known num_inputs to broadcast it to one
                    default_dimensions = py.broadcast(default_dimensions, num_inputs)
                else:
                    # If it's already a list, make sure it has the right length
                    assert len(default_dimensions) == num_inputs, \
                        _string_stamper("The provided `default_dimensions` implies there are {} inputs, whereas the "
                                        "expected number of inputs is {}.".format(len(default_dimensions), num_inputs))
                # Get final dimensions list.
                dimensions = [dimension if dimension is not None else default_dimension
                              for dimension, default_dimension in zip(dimensions, default_dimensions)]
                # Sanity check to be extra sure.
                assert None not in dimensions
            else:
                # So dimensions isn't a list, so it's either None or an integer.
                assert isinstance(dimensions, (None, int)), \
                    _string_stamper("The `dimensions` argument must either be a list, integer or None.")
                # If dimensions is just a number, we know nothing, if we allow the user to assume proper broadcasting.
                # In this case, use default value
                num_inputs = default_num_inputs
                # At this point, dimensions can also be None. If this is the case, substitute in the default_dimensions
                dimensions = default_dimensions if dimensions is None else dimensions
                # Until this point, we were sure dimensions wasn't a list. But we don't know whether default_dimensions
                # was one, so we need another check.
                if isinstance(dimensions, list):
                    # Ok, so default_dimensions is a list, as is dimensions. Make sure the length checks out
                    assert len(dimensions) == num_inputs, \
                        _string_stamper("Infered number of inputs from the given `dimensions` (default or otherwise) "
                                        "is inconsistent with the inferred number of inputs. The former is {}, "
                                        "while the latter is {}.".format(len(dimensions), num_inputs))
                else:
                    # dimensions still isn't a list, so we make it one. This defines the broadcasting behaviour.
                    dimensions = py.broadcast(dimensions, num_inputs)
                    # Now we need to guarantee that default_dimensions is a list as well
                    if isinstance(default_dimensions, list):
                        # Ok, so default_dimensions is already a list. Make sure it's consistent with num_inputs
                        assert len(default_dimensions) == num_inputs, \
                            _string_stamper("The given `default_dimensions` (= {}) is not consistent with the "
                                            "expected number of inputs (= {}). Is the given `default_dimensions` "
                                            "consistent with the given `default_num_inputs`?")
                    else:
                        # So default_dimensions is not a list, so we make it one
                        default_dimensions = py.broadcast(default_dimensions, num_inputs)

            # Get final dimensions list. Look for None's in there and replace with defaults
            dimensions = [dimension if dimension is not None else default_dimension
                          for dimension, default_dimension in zip(dimensions, default_dimensions)]
            # Sanity check to be extra sure.
            assert None not in dimensions, _string_stamper("Sanity check failed. Well, shit.")

        else:
            # If num_inputs is given, check whether it's consistent with dimensions
            if isinstance(dimensions, list):
                assert len(dimensions) == num_inputs, \
                    _string_stamper("The given `dimensions` is a list of {} elements, i.e. that many inputs. "
                                    "This is not consistent with the given `num_inputs` (number of inputs), "
                                    "which is {}.".format(len(dimensions), num_inputs))
            else:
                # num_inputs is given, but dimensions is not a list. So let's make it one!
                dimensions = py.broadcast(dimensions, num_inputs)

        # Cobble together an input shape given what we know
        known_input_shape = py.delistlistoflists([[None for _ in range(_dimension_to_tensor_dimension[dimension])]
                                                  for dimension in dimensions])
    else:
        # Input shape is known (well, atleast the signature). Time for sanity checks.
        derived_num_inputs = len(known_input_shape) if py.islistoflists(known_input_shape) else 1
        derived_tensor_dimensions = map(len, known_input_shape) if py.islistoflists(known_input_shape) \
            else len(known_input_shape)
        derived_dimensions = map(lambda x: _tensor_dimension_to_dimension[x], derived_tensor_dimensions) \
            if isinstance(derived_tensor_dimensions, list) \
            else _tensor_dimension_to_dimension[derived_tensor_dimensions]

        # Check if the derivced num_input is consistent with num_inputs if given
        if num_inputs is not None:
            assert num_inputs == derived_num_inputs, \
                _string_stamper("The given number of inputs `num_inputs` (= {}) is not consistent with "
                                "the one derived from the given input shape `known_input_shape` "
                                "(= {}).".format(num_inputs, derived_num_inputs))
        else:
            # Set num_inputs
            num_inputs = derived_num_inputs
        # Check if the derived dimensions is consistent with the given dimensions (if given)
        if dimensions is not None:
            assert all([dimension == derived_dimension for dimension, derived_dimension
                        in zip(py.broadcast(dimensions, py.smartlen(derived_dimensions)),
                               py.obj2list(derived_dimensions))]), \
                _string_stamper("Given `dimensions` (= {}) is not consistent "
                                "with the one derived from the given input shape "
                                "`known_input_shape`.".format(dimensions, derived_dimensions))
        else:
            dimensions = derived_dimensions

    if num_features_in is not None:
        if py.islistoflists(known_input_shape):
            # Broadcast num_features_in
            num_features_in = py.broadcast(num_features_in, len(known_input_shape)) \
                if py.smartlen(num_features_in) != len(known_input_shape) else num_features_in

            # Loop over input indices and set num_features_in in known_input_shape
            # (no need to worry about dimensions - that's why I'm starting to love the b01c format)
            for input_idx in range(len(num_features_in)):
                if known_input_shape[input_idx][-1] is None:
                    known_input_shape[input_idx][-1] = num_features_in[input_idx]
                else:
                    assert known_input_shape[input_idx][-1] == num_features_in[input_idx], \
                        _string_stamper("Given number of input features (= {}) for input "
                                        "indexed {} is not consistent with its expected "
                                        "shape {} (= {}).".format(num_features_in[input_idx],
                                                                  input_idx, known_input_shape[input_idx],
                                                                  known_input_shape[input_idx][-1]))

        else:
            # Expecting just one input - easy peasy
            assert known_input_shape[-1] is None or known_input_shape[-1] == num_features_in, \
                _string_stamper("Given number of input features ({}) is not consistent with the given input_shape "
                                "(expecting {} input features).".format(num_features_in, known_input_shape[-1]))
            known_input_shape[-1] = num_features_in

    return known_input_shape


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


class DictList(OrderedDict):
    """
    This class brings some of the list goodies to OrderedDict (including number indexing), with the caveat that
    keys are only allowed to be strings.
    """

    def __init__(self, item_list, **kwds):
        # Try to make item_list compatible
        item_list = self._make_compatible(item_list)
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

    def _make_compatible(self, obj):
        # Get compatibility code
        code = self._is_compatible(obj, find_key_conflicts=False)

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


class ParameterCollection(OrderedDict):
    """Class to collect parameters of a layer."""
    # TODO: Can of worms for another day.
    pass


if __name__ == '__main__':
    pass
