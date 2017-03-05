import pyutils2 as py2


# ---- DECORATORS


def changes_graph(function):
    """
    Decorator to let the LayerGraph know that this method changes the graph, i.e. the caches
    are invalidated.
    """
    def _graph_changing_function(cls, *args, **kwargs):
        cls.graph_has_changed()
        return function(*args, **kwargs)
    return _graph_changing_function


# ---- NAMING


def find_a_name(layer_or_model, all_names, given_name=None, _string_stamper=None):
    _string_stamper = (lambda x: x) if _string_stamper is None else _string_stamper
    # Check if layer_or_model has a user defined name
    if layer_or_model.name_is_user_defined:
        inferred_name = layer_or_model.name
        # Make sure the given name (if provided) is the same as the inferred name
        assert given_name is None or given_name == inferred_name, \
            _string_stamper("Given name '{}' is not consistent " \
                            "with inferred name '{}'.".format(given_name, inferred_name))
        # If name is indeed user defined, make sure its unique
        assert inferred_name not in all_names, \
            _string_stamper("The layer name '{}' is not unique.".format(inferred_name))
        # Done; inferred name is the name
        return inferred_name
    else:
        # layer or model has no name, so we need to find one.
        return py2.autoname_layer_or_model(layer_or_model, _string_stamper=_string_stamper)
