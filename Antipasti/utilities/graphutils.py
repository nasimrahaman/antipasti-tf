

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


def find_a_name(given_name=None, all_names=None, layer_or_model=None):
    # Check if layer_or_model has a user defined name
    if layer_or_model.name_is_user_defined:
        inferred_name = layer_or_model.name
        assert given_name is None or given_name == inferred_name, \
            "Given name '{}' is not consistent " \
            "with inferred name '{}'.".format(given_name, inferred_name)
    # TODO
