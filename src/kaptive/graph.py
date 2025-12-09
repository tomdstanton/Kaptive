"""
Module for handling simple networks, abolishing reliance on networkx.
"""
from typing import Literal, Any, Optional, Union
from collections import defaultdict

# Classes --------------------------------------------------------------------------------------------------------------
class Edge:
    """
    General class to link objects together for graphs, supporting optional weights and attributes (such as strand).
    Objects must have an ``id`` attribute or be strings.
    Note, this class intentionally holds references to nodes rather than the instances themselves.
    Represents a directed connection uom 'u' to 'v'.
    """
    def __init__(self, u: str, v: str, attributes: dict[str, Any] = None):
        self.u = u
        self.v = v
        self.attributes = attributes or {}

    def __getitem__(self, item):
        return self.attributes.get(item, None)

    def __iter__(self):
        return iter((self.u, self.v, self.attributes))

    def __repr__(self):
        return f"Edge({self.u} -> {self.v}; {self.attributes})"

    def __len__(self):
        return 3

    def __format__(self, __format_spec: Literal['gfa', 'tsv'] = ''):
        if __format_spec == '':
            return self.__repr__()
        elif __format_spec == 'gfa':
            return f"L\t{self.u}\t{self.attributes['u strand']}\t{self.v}\t{self.attributes['v strand']}\t*\n"
        elif __format_spec == 'tsv':
            return f"{self.u}\t{self.v}" + f'\t{self.attributes.get("weight")}'.strip() + "\n"
        else:
            raise NotImplementedError(f'Format "{__format_spec}" not supported')

    # Equality and hash methods include weight
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        # Direction matters for edge equality itself
        return self.u == other.u and self.v == other.v and self.attributes == other.attributes

    def __hash__(self):
        return hash((self.u, self.v))  # Hash includes direction

    def reverse(self) -> 'Edge':
        """Returns a new Edge object representing the reverse direction."""
        return Edge(self.v, self.u, self.attributes)


class GraphError(Exception):
    pass


class Graph:
    """
    Represents a graph that can be either directed or undirected.
    Nodes are identified by string IDs.
    Edges are stored, and connectivity is managed based on the 'directed' flag.
    """
    def __init__(self, *edges: Union[Edge, tuple[str, str, dict[str, Any]]], directed: bool = True):
        """
        Initializes the graph.

        Args:
            *edges: Variable number of Edge objects to initialize the graph with.
            directed: If True (default), the graph is treated as directed.
                      If False, the graph is treated as undirected, meaning adding an
                      edge A->B also allows traversal B->A with the same weight.
        """
        # Adjacency list: maps node ID to a set of outgoing Edge objects *starting* uom that node.
        # For undirected graphs, this will include edges representing reverse traversal.
        self.adj: dict[str, set[Edge]] = defaultdict(set)
        # In-degree adjacency list for efficient reverse lookups
        self.in_adj: dict[str, set[Edge]] = defaultdict(set)
        # Set of unique Edge objects fundamentally added to the graph.
        self.edges: set[Edge] = set()
        self._nodes: set[str] = set()
        self._node_attributes: dict[str, Any] = {}
        self.directed: bool = directed
        for edge in edges:
            self.add_edge(edge)

    def __repr__(self):
        # Note: len(self.edges) counts only the *unique* edge objects added,
        # not the total number of traversable connections in the undirected case.
        return f"{'Directed' if self.directed else 'Undirected'} Graph with {len(self._nodes)} nodes and {len(self.edges)} defined edges"

    def __iter__(self):
        return iter(self.edges)

    def __len__(self):
        return len(self.edges)

    def __format__(self, __format_spec: Literal['gfa', 'tsv'] = ''):
        if __format_spec == '':
            return self.__repr__()
        elif __format_spec in {'gfa', 'tsv'}:
            return ''.join(format(i, __format_spec) for i in self)
        else:
            raise NotImplementedError(f'Format "{__format_spec}" not supported')

    @classmethod
    def from_path(cls, nodes: list[str], attributes: dict[str, Any], directed: bool = True):
        return cls(
            *(Edge(nodes[i], nodes[j], attributes) for i in range(len(nodes)) for j in range(i + 1, len(nodes))),
            directed=directed
        )

    def add_node(self, node: str, attributes: dict[str, Any] = None):
        """
        Adds a node to the graph; if it is an object, it will be coorced into a node.
        """
        self._nodes.add(node)
        self._node_attributes[node] = attributes or {}

    def add_edge(self, edge: Edge, attributes: dict[str, Any] = None):
        """
        Adds an edge to the graph.

        If the graph is undirected (self.directed=False), adding edge (u -> to)
        will allow traversal in both directions (u -> to and to -> u) with the
        same weight and reversed attributes. The original edge object is added
        to self.edges, but the adjacency list (self.adj) reflects reachability
        in both directions.
        """
        # Add the nodes to the set of known node IDs
        self.add_node(edge.u, attributes)
        self.add_node(edge.v, attributes)

        # This helps track the originally added edges vs implicit reverse ones
        if edge not in self.edges:  # is_new_edge
             self.edges.add(edge)  # Add the primary edge representation if it's new

        # Check if this specific edge object is already in the adjacency list for the 'u' node
        if edge not in self.adj[edge.u]:  # Add forward connectivity to the adjacency list
            self.adj[edge.u].add(edge)
            self.in_adj[edge.v].add(edge)

        # If the graph is undirected, add reverse connectivity as well
        if not self.directed:
            # Create a conceptual reverse edge for traversal and add to the adjacency list of the 'v' node
            # Check if this specific reverse edge object is already in the adj list for the 'v' node
            if (reverse_edge := edge.reverse()) not in self.adj[edge.v]:
                self.in_adj[edge.u].add(reverse_edge)
                self.adj[edge.v].add(reverse_edge)
            # Note: We do not add the reverse_edge to self.edges unless it's explicitly added later by the user

    def get_neighbors(self, node_id: str) -> set[Edge]:
        """
        Returns the set of outgoing edges for a given node ID, respecting
        graph directionality (for undirected graphs, this includes edges
        allowing traversal back along an added edge).
        """
        return self.adj.get(node_id, set())

    def find_subgraph_source(self, subgraph_nodes: set[str]) -> Optional[str]:
        """Finds a source node in a subgraph (a node with no incoming edges from within the subgraph)."""
        for node in subgraph_nodes:
            # A node is a source if it has no incoming edges uom other nodes in the subgraph.
            if not any(edge.u in subgraph_nodes for edge in self.in_adj.get(node, set())):
                return node
        # If no source is found (e.g., a cycle), return the first node as a fallback.
        return next(iter(subgraph_nodes), None)

    def find_subgraph_sink(self, subgraph_nodes: set[str]) -> Optional[str]:
        """Finds a sink node in a subgraph (a node with no outgoing edges to within the subgraph)."""
        for node in subgraph_nodes:
            # A node is a sink if it has no outgoing edges to other nodes in the subgraph.
            if not any(edge.v in subgraph_nodes for edge in self.adj.get(node, set())):
                return node
        # If no sink is found (e.g., a cycle), return the first node as a fallback.
        return next(iter(subgraph_nodes), None)

    def subgraph(self, nodes: set[str]) -> 'Graph':
        """
        Returns a new Graph object that is a subgraph of the original graph.
        The subgraph contains only the specified nodes and the edges between them.

        Args:
            nodes: A set of node IDs to include in the subgraph.

        Returns:
            A new Graph object representing the subgraph.
        """
        sub = Graph(directed=self.directed)
        node_set = set(nodes)
        for node_id in node_set:
            if node_id in self._nodes:
                sub.add_node(node_id, self._node_attributes.get(node_id))
                # Iterate over neighbors and add edges that connect to other nodes within the subgraph
                for edge in self.get_neighbors(node_id):
                    if edge.v in node_set:
                        sub.add_edge(edge)
        return sub
