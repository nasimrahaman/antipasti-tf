import pyutils2 as py2
import networkx as nx
from collections import OrderedDict


# --- CORE


class NetworkGraph(nx.DiGraph):
    """A NetworkX DiGraph, except that node and edge ordering matters."""
    node_dict_factory = OrderedDict
    adjlist_dict_factory = OrderedDict


class ConnectivitySpec(object):
    """
    Specifies an arbitrary connection between (groups of) layers in a `NetworkGraph`.
    This warrants its own class because any of the layers may have an arbitrary number of
    inputs and outputs.
    """
    # TODO
    pass


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


def find_a_name(layer, all_names, given_name=None, _string_stamper=None):
    _string_stamper = (lambda x: x) if _string_stamper is None else _string_stamper
    # First priority goes to given_name
    if given_name is not None:
        return py2.autoname_layer_or_model(given_name=given_name)

    # A name is not given. Check if layer has a user defined name
    if layer.name_is_user_defined:
        inferred_name = layer.name
        # If name is indeed user defined, make sure its unique
        assert inferred_name not in all_names, \
            _string_stamper("The layer name '{}' is not unique.".format(inferred_name))
        # Done; inferred name is the name
        return inferred_name
    else:
        # layer or model has no name, so we need to find one.
        return py2.autoname_layer_or_model(layer, given_name=given_name,
                                           _string_stamper=_string_stamper)


def split_address_to_node_name_and_port(address, separator='::'):
    if separator not in address:
        # address is the node_name, and port is 0
        return address, 0
    else:
        return tuple(address.split('::'))
