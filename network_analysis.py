"""
File: network_analysis.py

Author: Tomás Maia
Date: 2025-02-21

Description:
This script contains various functions for analyzing network topologies, generating traffic demands, 
and visualizing network structures. The functions use the NetworkX library for graph-related operations 
and Matplotlib for visualizing network topologies. The script can load network graphs from .graphml files, 
compute topological characteristics, perform routing, and display results, including link loads and capacity.

Functions:
- calculate_topological_characteristics: Calculates and optionally prints the network's topological features.
- display_routing_results: Displays routing statistics, link loads, and wavelength usage.
- generate_random_demand: Generates random traffic demands with paths based on precomputed shortest paths.
- load_generated_graphs: Loads .graphml files from a given folder into NetworkX graph objects.
- load_networks_from_folders: Loads .graphml files from multiple folders.
- filter_random_networks: Filters graphs to limit the number of networks per node count.
- visualize_random_network: Visualizes the network topology with node labels and edge distances.

Dependencies:
- numpy
- networkx
- matplotlib
- random
- os

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import os

from path_computation import path_distance

# Function to calculate and display or return network characteristics
def calculate_topological_characteristics(G, verbose=True):
    """
    Calculate and optionally print network topological characteristics.

    Parameters:
    - G: NetworkX graph.
    - verbose: If True, prints the characteristics. If False, returns them as a dictionary.

    Returns:
    - Dictionary of network characteristics (if verbose is False).
    """
    link_lengths = [d['weight'] for (u, v, d) in G.edges(data=True)]
    node_degrees = [deg for _, deg in G.degree()]
    
    characteristics = {
        "num_nodes": G.number_of_nodes(),
        "num_links": G.number_of_edges(),
        "min_link_length": min(link_lengths),
        "max_link_length": max(link_lengths),
        "avg_link_length": np.mean(link_lengths),
        "var_link_length": np.var(link_lengths),
        "min_node_degree": min(node_degrees),
        "max_node_degree": max(node_degrees),
        "avg_node_degree": np.mean(node_degrees),
        "var_node_degree": np.var(node_degrees),
        "network_diameter": nx.diameter(G, weight="weight"),
        "algebraic_connectivity": nx.algebraic_connectivity(G, weight="weight"),
    }

    if verbose:
        print("----- NETWORK TOPOLOGICAL CHARACTERISTICS -----")
        for i, (key, value) in enumerate(characteristics.items(), start=1):
            print(f"{i} - {key.replace('_', ' ').title()}: {value}")

    return characteristics

# Additional function to calculate and display routing results and link loads
def display_routing_results(G, available_wavelengths, successful_connections, blocked_connections, Cnet, num_channels):
    """
    Displays routing results, link loads, and network statistics.

    Parameters:
    - G: NetworkX graph.
        The graph object representing the network.
    - available_wavelengths: dict.
        A dictionary mapping edges to lists indicating available wavelengths.
    - successful_connections: list of tuples.
        Each tuple contains a source node, target node, path, and wavelength for a successful connection.
    - blocked_connections: list of tuples.
        Each tuple contains a source node, target node, and path for a blocked connection.
    - Cnet: float.
        The total network capacity in bits per second.
    - num_channels: int.
        The total number of channels in the network.

    Returns:
    - None.
        The function prints routing statistics, link load, wavelength usage, and capacity results.
    """

    # Calculate path length, hops, and load statistics
    path_lengths = [path_distance(G, demand[2]) for demand in successful_connections]
    avg_path_length = np.mean(path_lengths)
    hops = [len(demand[2]) - 1 for demand in successful_connections]
    avg_hops = np.mean(hops)

    # Link loads and wavelength utilization
    link_loads = {link: num_channels - sum(w) for link, w in available_wavelengths.items()}
    avg_link_load = np.mean(list(link_loads.values()))
    min_link_load = min(link_loads.values())
    max_link_load = max(link_loads.values())

    # Print routing statistics
    if verbose:
        print("\n----- ROUTING RESULTS -----")
        print(f"Minimum Path Length: {min(path_lengths)} km")
        print(f"Maximum Path Length: {max(path_lengths)} km")
        print(f"Average Path Length: {avg_path_length:.2f} km")
        print(f"Average Hops per Demand: {avg_hops:.2f}")
        print(f"Minimum Link Load: {min_link_load}")
        print(f"Maximum Link Load: {max_link_load}")
        print(f"Average Link Load: {avg_link_load:.2f}")

        # Blocking results
        blocking_probability = len(blocked_connections) / (len(successful_connections) + len(blocked_connections))
        print(f"Total Blocked Traffic Demands: {len(blocked_connections)}")
        print(f"Blocking Probability: {blocking_probability:.4f}")

        # Print blocked paths
        print("Blocked Paths:")
        for blocked in blocked_connections:
            source, target = blocked[:2]
            print(f"Source {source}, Destination {target}, Path: {blocked}")

        # Link load and wavelength usage
        print("\n----- LINK LOAD AND WAVELENGTHS -----")
        for (u, v), load in link_loads.items():
            used_wavelengths = [i + 1 for i, available in enumerate(available_wavelengths[(u, v)]) if not available]
            print(f"{u}-{v} -> Load: {load}, Wavelengths: {used_wavelengths}")

        # Print traffic demands with paths and wavelengths
        print("\n----- TRAFFIC DEMAND PATHS AND WAVELENGTHS -----")
        for demand in successful_connections:
            source, target, path, wavelength = demand
            print(f"{source}-{target} -> Path: {path}, Wavelength {wavelength + 1}")

        # Print capacity results
        print("\n----- CAPACITY RESULTS -----")
        avg_channel_capacity = Cnet / len(successful_connections) / 1e9 if successful_connections else 0
        print(f"Average Channel Capacity: {avg_channel_capacity:.2f} Gbps")
        print(f"Total Network Capacity: {Cnet / 1e12:.2f} Tbps")

def generate_random_demand(G, num_nodes, all_pairs_shortest_paths):
    """
    Generate a random traffic demand for progressive traffic, ensuring paths are retrieved correctly.

    Parameters:
    - num_nodes: Number of nodes in the network.
    - all_pairs_shortest_paths: Precomputed k-shortest paths for one direction of node pairs.

    Returns:
    - demand1: A tuple (source, target, paths, shortest_path_length).
    - demand2: A tuple (target, source, reversed_paths, shortest_path_length).
    """
    while True:
        # Randomly select source and target nodes
        source = random.randint(0, num_nodes - 1)
        target = random.randint(0, num_nodes - 1)

        # Ensure source and target are not the same
        if source != target:
            # Retrieve paths for the forward direction
            paths_forward = all_pairs_shortest_paths.get((str(source), str(target)))

            if paths_forward:
                # Reverse the forward paths for the reverse direction
                paths_reverse = [list(reversed(path)) for path in paths_forward]

                # Compute the shortest path length for the forward direction
                shortest_path_len = path_distance(G, paths_forward[0])

                # Structure the demands
                demand1 = (source, target, paths_forward, shortest_path_len)
                demand2 = (target, source, paths_reverse, shortest_path_len)
                return demand1, demand2

# Function to load graphs from the generated .graphml files
def load_generated_graphs(folder_path):
    """
    Loads all .graphml files from the specified folder into a list of NetworkX graphs.

    Parameters:
    - folder_path: Path to the folder containing the .graphml files.

    Returns:
    - A dictionary with graph names as keys and NetworkX graph objects as values.
    """
    graphs = {}
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".graphml"):
            graph_path = os.path.join(folder_path, file_name)
            G = nx.read_graphml(graph_path)
            graphs[file_name] = G

    return graphs

def load_networks_from_folders(folder_paths):
    """
    Loads all .graphml files from the specified folder into a list of NetworkX graphs.

    Parameters:
    - folder_paths: List of folder paths containing .graphml files.

    Returns:
    - A dictionary with graph names as keys and NetworkX graph objects as values.
    """
    all_graphs = {}
    for folder in folder_paths:
        if os.path.exists(folder):
            graphs = load_generated_graphs(folder)  # Reuse existing function for a single folder
            all_graphs.update(graphs)
        else:
            print(f"Warning: Folder '{folder}' does not exist. Skipping.")
    return all_graphs

def filter_random_networks(graphs, node_counts, max_networks_per_count):
    """
    Filters and limits the number of random networks for each node count.

    Parameters:
    - graphs: Dictionary of graphs (name -> graph).
    - node_counts: List of desired node counts to analyze.
    - max_networks_per_count: Maximum number of networks per node count.

    Returns:
    - A filtered dictionary of graphs with a limited number per node count.
    """
    filtered_graphs = {}
    for node_count in node_counts:
        matching_graphs = {name: G for name, G in graphs.items() if len(G.nodes) == node_count}
        selected_graphs = dict(list(matching_graphs.items())[:max_networks_per_count])  # Limit to max_networks_per_count
        filtered_graphs.update(selected_graphs)
    return filtered_graphs

def visualize_random_network(G, title="Random Network Topology", random = True):
    """
    Visualizes a network graph with node labels and edge distances.
    
    Parameters:
    - G: The network graph.
    - title: Title for the plot.
    """
    plt.figure(figsize=(12, 8))

    pos = nx.get_node_attributes(G, 'pos')

    if pos:
            pos = {
                node: tuple(map(float, p.strip('[]').split(',')))
                for node, p in pos.items()
            }

    # If no positions were set, generate a layout (this is the fallback)
    if not pos:
        pos = nx.spring_layout(G, seed=42)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    # Draw edges with weights
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title(title)
    plt.axis("off")
    plt.show()