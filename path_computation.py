'''
file: path_computation.py

Author: Tomás Maia
Date: February 2025

Description: This module provides functions to compute k-shortest paths in a 
network graph, calculate the total distance of a path, and generate k-shortest 
paths for all source-destination pairs.

'''

import networkx as nx
from itertools import islice

def k_shortest_paths(G, source, target, k=1, weight=None):
    """
    Compute k-shortest paths between source and target in a graph.

    Parameters:
    - G: The networkx graph.
    - source: Source node.
    - target: Target node.
    - k: Number of shortest paths to compute (default=2).
    - weight: Edge attribute to use as weight (default=None).

    Returns:
    - List of k shortest paths (each path is a list of nodes).
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def compute_k_shortest_paths(graph, k_paths, traffic_matrix=None):
    """
    Compute k-shortest paths for all source-destination pairs.
    If no traffic matrix is provided, assumes a full-mesh logical topology.

    Parameters:
    - graph: The networkx graph.
    - k_paths: Number of k-shortest paths to compute.
    - traffic_matrix: Traffic matrix with demands (optional).
      If None, assumes a full-mesh logical topology.

    Returns:
    - Dictionary of k-shortest paths for each source-target pair.
    """
    all_pairs_shortest_paths = {}
    nodes = list(graph.nodes)

    # Ensure nodes are treated as integers for consistent comparisons
    try:
        nodes = sorted(nodes, key=int)  # Sort nodes as integers
    except ValueError:
        raise ValueError("Graph nodes must be convertible to integers for sorting.")

    if traffic_matrix is None:
        # Default: Full-mesh logical topology
        traffic_matrix = {(i, j): 1 for i in nodes for j in nodes if int(i) < int(j)}

    for (source, target), demand in traffic_matrix.items():
        # Convert source and target to integers for consistent order
        if int(source) < int(target) and demand > 0:
            paths = k_shortest_paths(graph, source, target, k=k_paths, weight="weight")
            all_pairs_shortest_paths[(source, target)] = paths

    return all_pairs_shortest_paths

def path_distance(G, path):
    """
    Calculate the total distance of a path in a network.

    Parameters:
    - G (networkx.Graph): The network graph.
    - path (list): List of nodes representing the path.

    Returns:
    - float: Total distance of the path.
    """
    distance = 0
    for i in range(len(path) - 1):
        distance += G[path[i]][path[i+1]]['weight']
    return distance