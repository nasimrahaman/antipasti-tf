from ..models import Model
from ..utilities import graphutils as gutils
from ..legacy import pykit as py
from collections import OrderedDict

try:
    import networkx as nx
except ImportError:
    print("NetworkX not found!")
    raise


class LayerGraph(Model):
    """Class to implement arbitrary graphs of layers with NetworkX."""
    def __init__(self, graph=None, input_shape=None, name=None):
        """
        :type graph: networkx.DiGraph
        :param graph: The network as a networkx graph.

        :type input_shape: list or list of list
        :param input_shape: Input shape.

        :type name: str
        :param name: Name of the model.
        """
        # Initialize superclass
        super(LayerGraph, self).__init__(name=name)

        # Properties
        self._graph = None
        self._graph_has_changed_since_the_last_update_of = {}
        self._caches = {}

        # Assignments
        self.graph = graph
        self.input_shape = input_shape

    @property
    def graph(self):
        # If graph is being used before being set, we define a graph and block all future
        # assignments
        if self._graph is None:
            self._graph = gutils.NetworkGraph()
        return self._graph

    @graph.setter
    def graph(self, value):
        if self._graph is None:
            if not isinstance(value, gutils.NetworkGraph):
                raise ValueError(self._stamp_string("Assigned graph must be a NetworkGraph, "
                                                    "found in Antipasti.utilities.graphutils. "
                                                    "Got an object of type {} instead.".
                                                    format(value.__class__.__name__)))
            self._graph = value
        else:
            raise RuntimeError(self._stamp_string("This model is already assigned a graph"))

    def write_to_cache(self, name, value):
        self._caches.update({name: value})

    def read_from_cache(self, name, default=None):
        self._caches.get(name, default)

    def has_graph_changed_since_the_last_update_of(self, what):
        return self._graph_has_changed_since_the_last_update_of.get(what, True)

    def graph_has_changed(self):
        """Tell the LayerGraph object that the graph has changed (invalidates all caches)."""
        for what in self._graph_has_changed_since_the_last_update_of.keys():
            self._graph_has_changed_since_the_last_update_of[what] = True

    @property
    def all_node_names(self):
        """Get names of all nodes in graph."""
        return self.graph.nodes

    @property
    def all_node_layers(self):
        """Get the associated layers of all nodes in graph."""
        return [node_attributes['layer'] for node_attributes in self.graph.node.values()]

    @property
    def node_name_to_layer_dict(self):
        return OrderedDict([(node_name, self.graph.node[node_name])
                            for node_name in self.all_node_names])

    @property
    def input_layers(self):
        """Input layers. These are the source nodes in the graph."""
        in_degrees = self.graph.in_degree()
        return [node for node in self.graph.nodes_iter() if in_degrees[node] == 0]

    @property
    def output_layers(self):
        """Output layers. These are the sink nodes in the graph."""
        out_degrees = self.graph.out_degree()
        return [node for node in self.graph.nodes_iter() if out_degrees[node] == 0]

    def is_layer_in_graph(self, layer):
        """
        Checks if a given layer is in the model graph.

        :type layer: Antipasti.layers.core.Layer or str
        :param layer: Layer to check. Can be a layer object or a string.
        """
        if isinstance(layer, str):
            return layer in self.all_node_names
        else:
            return layer in self.all_node_layers

    def get_node_name(self, layer):
        """
        Given a layer (as `str` or `Antipasti.layers.core.Layer`), this function finds its name(s)
        in the graph. Note that the same `Layer` object could have multiple corresponding nodes in
        the graph. If that's the case, this function returns a list with names of all such nodes.
        """
        if isinstance(layer, str):
            assert self.is_layer_in_graph(layer), \
                self._stamp_string("Layer '{}' was not found in graph.".format(layer))
            return layer
        else:
            node_name_to_layer_dict = self.node_name_to_layer_dict
            # One could have multiple nodes containing the same layer. This could be true e.g.
            # when weight sharing.
            node_names = py.delist([node_name
                                    for node_name, node_layer in node_name_to_layer_dict.items()
                                    if layer is node_layer])
            return node_names

    def get_layer(self, node_name):
        assert isinstance(node_name, str), self._stamp_string("`node_name` must be a string.")
        return self.graph.node[node_name]['layer']

    def _add_connection(self, source_to_target_dict):
        # TODO: Finish ConnectivitySpec first.
        pass

    def add(self, layer_or_graph, name=None, previous_=None):
        # TODO: Finish add_layer and add_graph first
        pass

    def add_layer(self, layer, name=None, previous_=None):
        """
        Add layer to graph.

        :type layer: Antipasti.layers.core.Layer
        :param layer: Layer to be added.

        :type name: str
        :param name: Name of the layer. Only if the layer was named previously, this argument must
                     either be None or the same as `layer.name`. The name will be postfixed by the
                     id of the given `layer`.

        :type previous_: str or Antipasti.layers.core.Layer or Antipasti.models.graph.LayerGraph or
                         list
        :param previous_: Previous layer(s). Can be (a list of) strings or `Layer`s or
                          `LayerGraph`s.
        """
        # Get layer name
        node_name = gutils.find_a_name(layer=layer, all_names=self.all_node_names,
                                       given_name=name)
        # Add layer to graph
        self.graph.add_node(node_name, layer=layer)
        # Add nodes before and after
        # TODO Finish _add_connection first
        if previous_ is not None:
            # Get name of the previous layer
            # Add layer to the previous
            pass
        else:
            # This is when we add the layer to the entire graph, i.e. all output nodes are connected
            pass
        pass

    def add_graph(self, graph, name=None, previous_=None, next_=None):
        # TODO Finish add_layer first
        pass
