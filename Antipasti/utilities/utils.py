__author__ = "Nasim Rahaman"

from contextlib2 import contextmanager

from .pyutils2 import DictList, get_parameter_tag
from .. import backend as A
from ..legacy import pykit as py


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
        # TODO Integrate context supermanager here
        output = A.call_in_managers(cls.context_managers)(forward_function)(cls, input=input)
        # Set flag to indicate that the layer has been fedforward
        cls._is_fedforward = True
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
                assert isinstance(dimensions, (type(None), int)), \
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
            num_features_in = py.broadcast(num_features_in, len(known_input_shape))

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

    return py.delistlistoflists(known_input_shape)


def get_layer_xy_placeholders(input_shape=None, output_shape=None, device=None, variable_scope=None,
                              context_managers=None, layer_id=None):
    """
    Every Antipasti `Layer` should have 'x' (input) and 'y' (output) attributes (tf.placeholder).
    This function generates them given the input and output shapes.
    """

    # Container for variables
    xy_variables = DictList([])

    # Fetch x variable
    if input_shape is not None:
        if not py.islistoflists(input_shape):
            xy_variables['x'] = A.placeholder(shape=input_shape, device=device, variable_scope=variable_scope,
                                              other_context_managers=context_managers,
                                              antipasti_name=(None if layer_id is None else
                                                              get_parameter_tag(layer_id, 'x')))
        else:
            xy_variables['x'] = [A.placeholder(shape=_input_shape, device=device, variable_scope=variable_scope,
                                               other_context_managers=context_managers,
                                               antipasti_name=(None if layer_id is None else
                                                               get_parameter_tag(layer_id, 'x{}'.format(_input_id))))
                                 for _input_id, _input_shape in enumerate(input_shape)]
            pass

    if output_shape is not None:
        if not py.islistoflists(output_shape):
            xy_variables['y'] = A.placeholder(shape=output_shape, device=device, variable_scope=variable_scope,
                                              other_context_managers=context_managers,
                                              antipasti_name=(None if layer_id is None else
                                                              get_parameter_tag(layer_id, 'y')))
        else:
            xy_variables['y'] = [A.placeholder(shape=_output_shape, device=device, variable_scope=variable_scope,
                                               other_context_managers=context_managers,
                                               antipasti_name=(None if layer_id is None else
                                                               get_parameter_tag(layer_id, 'y{}'.format(_output_id))))
                                 for _output_id, _output_shape in enumerate(output_shape)]
    return xy_variables


def compare_shapes(shape1, shape2, soft=True):
    """
    Function to compare shapes while accounting for unknown components (set to None).
    This function does not return whether the shapes are equal, but whether they 'could' be equal, i.e.
    whether they're compatible. Setting soft to True (False) would have this function (not) ignore None's.
    """

    # Define function to compare lists (barring None's)
    def _compare_lists(list1, list2):
        return len(list1) == len(list2) and \
               all([elem1 == elem2 if None not in [elem1, elem2] else soft for elem1, elem2 in zip(list1, list2)])

    # First test: shape1 and shape2 must both (not) be a list of lists
    shapes_are_equal = py.islistoflists(shape1) == py.islistoflists(shape2)
    # Second test: number of inputs must be equal
    shapes_are_equal = shapes_are_equal and (len(py.list2listoflists(shape1)) == len(py.list2listoflists(shape2)))
    # Third test: list comparisons
    shapes_are_equal = shapes_are_equal and all([_compare_lists(list1, list2) for list1, list2 in
                                                 zip(py.list2listoflists(shape1), py.list2listoflists(shape2))])
    # Done.
    return shapes_are_equal


def validate_shape(variable, expected_shape, soft=True, set_shape=False):
    """
    Validates variable shape. Can also work with multiple variables if `variable` is passed in as a list and
    `expected_shape` as a list of lists. Setting soft to True would result in soft validation (ignoring None's).
    If `set_shape` is set to True, the placeholder shape is set with the set_shape method (it has no effect
    if `variable` is not a placeholder, i.e. does not have a `set_shape` attribute).

    :type variable: list or any
    :param variable: Variable(s)

    :type expected_shape: list or list of list
    :param expected_shape: Expected shape(s)

    :type soft: bool
    :param soft: Whether to ignore None's while comparing shapes.

    :type set_shape: bool
    :param set_shape: Whether to set the shape of the variable to `expected_shape`

    :return: (A list of) True if shape(s) is (are) valid, False otherwise
    """
    _vars = py.obj2list(variable)
    _var_shapes = [A.shape(_var) for _var in _vars]
    _expected_shapes = py.list2listoflists(expected_shape)
    validation_results = []

    if len(_var_shapes) != len(_vars):
        raise ValueError("Cannot validate shape. Given list of variables has {} "
                         "elements, but the given list of shapes has {}."
                         .format(len(_vars), len(_var_shapes)))

    for _var_num in range(len(_vars)):
        shape_soft_ok = compare_shapes(_var_shapes[_var_num], _expected_shapes[_var_num], soft=True)
        shape_hard_ok = compare_shapes(_var_shapes[_var_num], _expected_shapes[_var_num], soft=False)
        # Set shape (if requested) if required shape is not hard ok but just soft ok
        if hasattr(_vars[_var_num], 'set_shape') and set_shape and not shape_hard_ok and shape_soft_ok:
            # Set shape here
            _vars[_var_num].set_shape(_expected_shapes[_var_num])
        # Add validation result to list
        validation_results.append(shape_soft_ok if soft else shape_hard_ok)

    # Delist and return
    return py.delist(validation_results)


def get_shape(variable):
    """Like backend.shape, but also works for lists of variables."""
    return py.delistlistoflists([A.shape(var) for var in py.obj2list(variable)])


class LayerContextSuperManagers(object):
    """
    Context manager to be used by Layer. Contains two context supermanagers, one for initializing parameters and
    the other for feeding forward (i.e. building the graph). This can be useful for e.g. synchronous training, where
    the parameters are initialized on the CPU with a certain set of context managers (i.e. tf.device('/cpu:0')) but
    fedforward on the GPU with another set of context managers (i.e. tf.device('/gpu:0')).
    """
    def __init__(self, initialize_csm=None, feedforward_csm=None, default_csm_name='initialize'):
        """
        :type initialize_csm: Antipasti.backend.ContextSuperManager
        :param initialize_csm: Context supermanager for variable initialization.

        :type feedforward_csm: Antipasti.backend.ContextSuperManager
        :param feedforward_csm: Context supermanager for feeding forward

        :type default_csm_name: str
        :param default_csm_name: Default context supermanager (can be 'initialize' or 'feedforward')
        """
        # Property containers
        self._default_csm_name = None
        self._default_csm = None
        self._ACCESSIBLE_ATTRIBUTES = ['device', 'variable_scope', 'other_context_managers']

        # Attrubute Assignment
        self.initialize_csm = initialize_csm
        self.feedforward_csm = feedforward_csm
        self.default_csm_name = default_csm_name

    @property
    def default_csm_name(self):
        return self._default_csm_name

    @default_csm_name.setter
    def default_csm_name(self, value):
        if value in ['initialize', 'init', 'i']:
            self._default_csm_name = 'initialize'
            self._default_csm = self.initialize_csm
        elif value in ['feedforward', 'ffd', 'f']:
            self._default_csm_name = 'feedforward'
            self._default_csm = self.feedforward_csm
        else:
            raise ValueError("Default csm name {} is not understood. "
                             "Try 'initialize' or 'feedforward' instead.".format(value))

    @contextmanager
    def manage(self, mode=None, **kwargs):
        if mode not in ['initialize', 'feedforward', None]:
            raise ValueError("Keyword `mode` must either be 'initialize' or 'feedforward'. Given: {}.".format(mode))

        assert not all([_csm is None for _csm in [self.initialize_csm, self.feedforward_csm]]), \
            "No context supermanager defined. Define either LayerContextSuperManagers.initialize_csm " \
            "or LayerContextSuperManagers.feedforward_csm before calling this method."

        # Find the right csm
        if mode == 'initialize':
            csm = self.initialize_csm
        elif mode == 'feedforward':
            csm = self.feedforward_csm
        else:
            if None not in [self.initialize_csm, self.feedforward_csm]:
                # None of the CSMs is None, so pick the default
                csm = self._default_csm
            else:
                # Exactly one of the two CSMs is None. Pick the one that isn't.
                csm = self.initialize_csm if self.initialize_csm is not None else self.feedforward_csm

        # CSM should now be defined. Get in context and yield
        with csm.manage(**kwargs) as scope:
            yield scope

    def set(self, what=None, value=None, for_='all'):
        # Parse value
        if isinstance(value, (list, tuple)):
            value_for_initialize_csm, value_for_feedforward_csm = value
        elif isinstance(value, dict):
            # We don't let the value string to default to None, because value = None can have a different meaning for
            # ContextSuperManager
            value_for_initialize_csm = value.get('initialize', 'x')
            value_for_feedforward_csm = value.get('feedforward', 'x')
        else:
            value_for_initialize_csm = value_for_feedforward_csm = value

        # Validate `what`
        if what not in self._ACCESSIBLE_ATTRIBUTES:
            raise ValueError("The keyword `what` of LayerContextSuperManagers.set "
                             "must be in {}. Got '{}' instead.".format(self._ACCESSIBLE_ATTRIBUTES, what))

        # Set value
        if for_ == 'all':
            if self.initialize_csm is not None and value_for_initialize_csm != 'x':
                setattr(self.initialize_csm, what, value_for_initialize_csm)
            if self.feedforward_csm is not None and value_for_feedforward_csm != 'x':
                setattr(self.feedforward_csm, what, value_for_feedforward_csm)
        elif for_ == 'initialize':
            if self.initialize_csm is not None and value_for_initialize_csm != 'x':
                setattr(self.initialize_csm, what, value_for_initialize_csm)
        elif for_ == 'feedforward':
            if self.feedforward_csm is not None and value_for_feedforward_csm != 'x':
                setattr(self.feedforward_csm, what, value_for_feedforward_csm)
        else:
            raise ValueError("The `for_` argument can either be 'all', "
                             "'initialize' or 'feedforward', got {} instead.".format(for_))

    def get(self, what):
        # Validate what
        if what not in self._ACCESSIBLE_ATTRIBUTES:
            raise ValueError("The keyword `what` of LayerContextSuperManagers.get "
                             "must be in {}. Got '{}' instead.".format(self._ACCESSIBLE_ATTRIBUTES, what))
        # Build dict and return
        what_dict = {}
        if self.feedforward_csm is not None:
            what_dict.update({'feedforward': getattr(self.feedforward_csm, what)})
        if self.initialize_csm is not None:
            what_dict.update({'initialize': getattr(self.initialize_csm, what)})
        return what_dict

    @property
    def device(self):
        return self.get('device')

    @device.setter
    def device(self, value):
        self.set(what='device', value=value, for_='all')

    @property
    def variable_scope(self):
        return self.get('variable_scope')

    @variable_scope.setter
    def variable_scope(self, value):
        self.set(what='variable_scope', value=value, for_='all')

    @property
    def other_context_managers(self):
        return self.get('other_context_managers')

    @other_context_managers.setter
    def other_context_managers(self, value):
        self.set(what='other_context_managers', value=value, for_='all')


def get_layer_context_manager(**kwargs):
    initialize_csm_kwargs = {key[len('initialize_'):]: val
                             for key, val in kwargs.items() if key.startswith('initialize_')}
    feedforward_csm_kwargs = {key[len('feedforward_'):]: val
                              for key, val in kwargs.items() if key.startswith('feedforward_')}

    # Build csms
    initialize_csm = A.ContextSuperManager(**initialize_csm_kwargs) if initialize_csm_kwargs else None
    feedforward_csm = A.ContextSuperManager(**feedforward_csm_kwargs) if feedforward_csm_kwargs else None

    # Build Layer manager
    layer_context_manager = LayerContextSuperManagers(initialize_csm=initialize_csm, feedforward_csm=feedforward_csm)
    # Done.
    return layer_context_manager


if __name__ == '__main__':
    pass
