from ..models import Model

try:
    import networkx as nx
except ImportError:
    print("NetworkX not found!")
    raise


class LayerGraph(Model):
    """Class to implement arbitrary graphs of layers with NetworkX."""
    def __init__(self, graph=None, input_shape=None, name=None):
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
        return self._graph

    @graph.setter
    def graph(self, value):
        # TODO
        pass

    def write_to_cache(self, name, value):
        self._caches.update({name: value})

    def read_from_cache(self, name, default=None):
        self._caches.get(name, default=default)

    def has_graph_changed_since_the_last_update_of(self, what):
        return self._graph_has_changed_since_the_last_update_of.get(what, True)

    def graph_has_changed(self):
        """Tell the LayerGraph object that the graph has changed (invalidates all caches)."""
        for what in self._graph_has_changed_since_the_last_update_of.keys():
            self._graph_has_changed_since_the_last_update_of[what] = True

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

    def add(self, layer_or_graph, name=None, previous_=None, next_=None):
        # TODO
        pass

    def add_layer(self, layer, name=None, previous_=None, next_=None):
        """
        Add layer to graph.

        :type layer: Antipasti.layers.core.Layer
        :param layer: Layer to be added.

        :type name: str
        :param name: Name of the layer. Only if the layer was named previously, this argument must
                     either be None or the same as `layer.name`. If the name is not unique, it will
                     be postfixed with an index.

        :type previous_: str or Antipasti.layers.core.Layer or Antipasti.models.graph.LayerGraph or
                         list
        :param previous_: Previous layer(s). Can be (a list of) strings or `Layer`s or
                          `LayerGraph`s.

        :type next_: str or Antipasti.layers.core.Layer or Antipasti.models.graph.LayerGraph or list
        :param next_: Previous layer(s). Can be a list of strings or `Layer`s or `LayerGraph`s.
        """
        # TODO
        pass

    def add_graph(self, graph, name=None, previous_=None, next_=None):
        # TODO
        pass
