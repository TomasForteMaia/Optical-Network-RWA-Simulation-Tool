import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.metrics.pairwise import haversine_distances
from itertools import islice

import time # Test computation times

# ----------------------------------------- #
#       PATH COMPUTATION FUNCTIONS          #
# ----------------------------------------- #

def k_shortest_paths_with_tie_breaking(G, source, target, k=10, weight="weight"):
    """
    Compute k-shortest paths, breaking ties for paths with the same length by prioritizing paths with smaller edge weights.
    """
    # Generate k-shortest paths
    paths = list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    # Sort paths:
    # Primary criterion: Path length (sum of edge weights)
    # Secondary criterion: Lexicographical order of edge weights
    sorted_paths = sorted(
        paths,
        key=lambda path: (
            sum(G[u][v][weight] for u, v in zip(path[:-1], path[1:])),  # Total path weight
            [G[u][v][weight] for u, v in zip(path[:-1], path[1:])]      # Lexicographical tie-breaker
        )
    )

    return sorted_paths  # Return sorted paths

def k_shortest_paths(G, source, target, k=2, weight=None):
    """
    Computes the k shortest paths between a source and a target node in the network.

    Parameters:
    - G (networkx.Graph): The network graph.
    - source (str/int): The source node.
    - target (str/int): The target node.
    - k (int): Number of shortest paths to compute.
    - weight (str): Edge attribute to use as weight for path computation.

    Returns:
    - List of k shortest paths (each path is a list of nodes).
    """
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def path_distance(G, path):
    """
    Computes the total weight (distance) of a given path.

    Parameters:
    - G (networkx.Graph): The network graph.
    - path (list): List of nodes representing the path.

    Returns:
    - float: Total weight of the path.
    """
    return sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))

def compute_paths_for_traffic_matrix(G, traffic_matrix=None, k=2, weight="weight"):
    """
    Computes k shortest paths only for (source, target) pairs where source < target.
    If no traffic matrix is provided, a full-mesh topology is assumed.

    Parameters:
    - G (networkx.Graph): The network graph.
    - traffic_matrix (dict, optional): Dictionary where keys are (source, target) tuples and values are demand counts.
    - k (int): Number of shortest paths to compute.
    - weight (str): Edge attribute to use as weight.

    Returns:
    - dict: Dictionary with (source, target) as keys and a list of k shortest paths as values.
    """
    all_pairs_shortest_paths = {}
    nodes = list(G.nodes)

    # Ensure nodes are treated as integers for consistent ordering
    try:
        nodes = sorted(nodes, key=int)
    except ValueError:
        raise ValueError("Graph nodes must be convertible to integers for sorting.")

    # Default: Full-mesh logical topology (1 demand per node pair)
    if traffic_matrix is None:
        traffic_matrix = {(i, j): 1 for i in nodes for j in nodes if int(i) < int(j)}

    for (source, target), demand in traffic_matrix.items():
        if int(source) < int(target) and demand > 0:
            paths = k_shortest_paths(G, source, target, k=k, weight=weight)
            all_pairs_shortest_paths[(source, target)] = paths

    return all_pairs_shortest_paths

def compute_paths_for_traffic_matrix_with_tie_breaking(G, traffic_matrix=None, k=2, weight="weight"):
    """
    Computes k shortest paths using tie-breaking for (source, target) pairs where source < target.
    If no traffic matrix is provided, a full-mesh topology is assumed.

    Parameters:
    - G (networkx.Graph): The network graph.
    - traffic_matrix (dict, optional): Dictionary where keys are (source, target) tuples and values are demand counts.
    - k (int): Number of shortest paths to compute.
    - weight (str): Edge attribute to use as weight.

    Returns:
    - dict: Dictionary with (source, target) as keys and a list of k shortest paths as values.
    """
    all_pairs_shortest_paths = {}
    nodes = list(G.nodes)

    # Ensure nodes are treated as integers for consistent ordering
    try:
        nodes = sorted(nodes, key=int)
    except ValueError:
        raise ValueError("Graph nodes must be convertible to integers for sorting.")

    # Default: Full-mesh logical topology (1 demand per node pair)
    if traffic_matrix is None:
        traffic_matrix = {(i, j): 1 for i in nodes for j in nodes if int(i) < int(j)}

    for (source, target), demand in traffic_matrix.items():
        if int(source) < int(target) and demand > 0:
            paths = k_shortest_paths_with_tie_breaking(G, source, target, k=k, weight=weight)
            all_pairs_shortest_paths[(source, target)] = paths

    return all_pairs_shortest_paths

# ----------------------------------------- #
#     TRAFFIC DEMAND ORDERING FUNCTION      #
# ----------------------------------------- #

def order_traffic_demands(G, all_pairs_shortest_paths, traffic_matrix, order_strategy="shortest-first"):
    """
    Orders traffic demands based on a specified strategy, ensuring bidirectional demands are consecutive.
    If the traffic matrix has multiple demands per pair, each demand is accounted for.

    Parameters:
    - G (networkx.Graph): The network graph.
    - all_pairs_shortest_paths (dict): Dictionary with (source, target) as keys and lists of paths as values.
    - traffic_matrix (dict): Dictionary with (source, target) pairs and their demand count.
    - order_strategy (str): Sorting strategy for traffic demands.

    Returns:
    - list: A sorted list of traffic demands in the form (source, target, paths, shortest_path_length).
    """
    traffic_demands = []

    # Step 1: Append each demand and its reverse direction (with correct demand count)
    for (source, target), paths in all_pairs_shortest_paths.items():
        shortest_path_len = path_distance(G, paths[0])
        demand_count = traffic_matrix.get((source, target), 1)  # Get actual demand count

        # Add all demands (if more than 1 demand exists, append multiple times)
        for _ in range(demand_count):
            traffic_demands.append((source, target, paths, shortest_path_len))  # Original direction
            reversed_paths = [list(reversed(path)) for path in paths]  # Reverse paths
            traffic_demands.append((target, source, reversed_paths, shortest_path_len))  # Reverse demand

    # Step 2: Sort based on the chosen strategy while keeping bidirectional demands together
    if order_strategy == "shortest-first":
        traffic_demands.sort(key=lambda x: x[3])
    elif order_strategy == "longest-first":
        traffic_demands.sort(key=lambda x: x[3], reverse=True)
    elif order_strategy == "random":
        # Shuffle bidirectional pairs together
        unique_pairs = list(set((min(d[0], d[1]), max(d[0], d[1])) for d in traffic_demands))
        random.shuffle(unique_pairs)
        shuffled_traffic_demands = []
        for src, dst in unique_pairs:
            for demand in traffic_demands:
                if {demand[0], demand[1]} == {src, dst}:
                    shuffled_traffic_demands.append(demand)
        traffic_demands = shuffled_traffic_demands
    else:
        raise ValueError("Unknown traffic sorting strategy.")

    return traffic_demands
   
# ----------------------------------------- #
#      WAVELENGTH ASSIGNMENT FUNCTION       #
# ----------------------------------------- #

def assign_wavelength(path, available_wavelengths, channels, assignment_strategy="first-fit"):
    """
    Assign a wavelength to a path based on the specified strategy if possible.

    Parameters:
    - path: The path for the demand.
    - available_wavelengths: A dictionary tracking the available wavelengths on each link.
    - channels: Total number of channels (wavelengths).
    - assignment_strategy: Strategy to assign wavelengths. Options: "first-fit", "most-used", "least-used", "random".

    Returns:
    - assigned_wavelength: The wavelength assigned to the path, or None if no wavelength is available.
    """

    # Track available wavelengths for the entire path
    path_wavelength_availability = [True] * channels  # Start with all wavelengths available

    for i in range(len(path) - 1):
        link = (path[i], path[i+1])

        # Combine the wavelength availability across the entire path
        link_wavelengths = available_wavelengths[link]
        path_wavelength_availability = [
            path_wavelength_availability[j] and link_wavelengths[j]
            for j in range(channels)
        ]

    # Determine wavelength order based on the assignment strategy
    if assignment_strategy == "first-fit":
        wavelength_order = range(channels)  # Default order by index (first-fit)
    elif assignment_strategy == "most-used":
        # Count wavelength usage across all links, sort by descending usage
        wavelength_usage_counts = [sum(1 for link in available_wavelengths if not available_wavelengths[link][w]) for w in range(channels)]
        wavelength_order = sorted(range(channels), key=lambda w: wavelength_usage_counts[w], reverse=True)
    elif assignment_strategy == "least-used":
        # Count wavelength usage across all links, sort by ascending usage
        wavelength_usage_counts = [sum(1 for link in available_wavelengths if not available_wavelengths[link][w]) for w in range(channels)]
        wavelength_order = sorted(range(channels), key=lambda w: wavelength_usage_counts[w])
    elif assignment_strategy == "random":
        wavelength_order = list(range(channels))
        random.shuffle(wavelength_order)  # Shuffle wavelengths randomly
    else:
        raise ValueError("Unknown wavelength assignment strategy.")

    # Find the first available wavelength across the whole path
    for w in wavelength_order:
        if path_wavelength_availability[w]:
            # Mark the wavelength as used on all links in the path
            for i in range(len(path) - 1):
                link = (path[i], path[i+1])
                available_wavelengths[link][w] = False  # Mark this wavelength as unavailable
            return w  # Return the assigned wavelength

    return None  # No wavelength available

# ----------------------------------------- #
#   HANDLE TRAFFIC & WAVELENGTH ALLOCATION #
# ----------------------------------------- #

def handle_traffic_demands(traffic_demands, available_wavelengths, channels, k, wavelength_assignment_strategy="first-fit"):
    """
    Assigns wavelengths to traffic demands using k-shortest paths.

    Parameters:
    - traffic_demands (list): List of (source, target, paths, shortest_path_length, demand_count).
    - available_wavelengths (dict): Dictionary tracking wavelength availability.
    - channels (int): Total number of available wavelengths.
    - k (int): Number of paths to consider for each demand.
    - wavelength_assignment_strategy (str): Strategy for wavelength assignment.

    Returns:
    - tuple: Lists of successful and blocked connections.
    """
    successful_connections = []
    blocked_connections = []

    for demand in traffic_demands:
        source, target, paths, _ = demand  # Extract source, target, and the k-shortest paths

        wavelength_assigned = False

        # Try each of the k paths for this demand
        for path_index, path in enumerate(paths[:k]):
            # Try to assign a wavelength for this path
            assigned_wavelength = assign_wavelength(path, available_wavelengths, channels, wavelength_assignment_strategy)
            
            if assigned_wavelength is not None:
                # Wavelength was successfully assigned
                # print(f"Assigned wavelength {assigned_wavelength} for demand from {source} to {target} using path {path}")

                # Store the successful connection
                successful_connections.append((source, target, path, assigned_wavelength))
                wavelength_assigned = True
                break  # Exit the loop since the demand was satisfied

        if not wavelength_assigned:
            # If no wavelength could be assigned on any of the k paths, block the demand
            # print(f"Blocked demand from {source} to {target}.")
            blocked_connections.append((source, target, paths[0]))

    return successful_connections, blocked_connections

def visualize_graph(G, title="Network Topology"):
    """
    Visualizes a network graph with node labels and edge distances.
    
    Parameters:
    - G: The network graph.
    - title: Title for the plot.
    """
    plt.figure(figsize=(12, 8))

    pos = nx.get_node_attributes(G, 'pos')
    if not pos:  # If 'pos' attribute is missing, generate a layout
        pos = nx.spring_layout(G, seed=42)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    # Draw edges with weights
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    
    plt.title(title)
    plt.axis("off")
    plt.show()

# Function to create a ring network with a given number of nodes
def create_ring_network(num_nodes, link_distance=1):
    """
    Creates a ring network graph with a specified number of nodes.
    
    Parameters:
    - num_nodes: The number of nodes in the ring network.
    - link_distance: The distance assigned to each link in km.
    
    Returns:
    - G: A NetworkX graph representing the ring network.
    """
    G = nx.Graph()
    
    # Add nodes and edges to form a ring
    for i in range(num_nodes):
        if(i + 1 == num_nodes):
            G.add_edge(i + 1, 1, weight=link_distance)
        else:
            G.add_edge(i + 1, i + 2, weight=link_distance)  # Wrap around to form a ring
    
    return G

# Function to define the COST239, NSFNET, and UBN networks based on Appendix A
def define_network(topology_name):
    G = nx.Graph()
    if topology_name == "CESNET":
        # Nodes and links with distances (COST239)
        nodes = [i for i in range(1, 8)]  # Use integers for nodes
        links = [
            (1, 2, 226.07), (1, 3, 334.4), (1, 7, 274.08),
            (2, 3, 315.98),
            (3, 4, 425.25),
            (4, 5, 378.51), (4, 6, 173.75),
            (5, 6, 212.79),
            (6, 7, 330.72),
        ]
    elif topology_name == "COST239":
        # Nodes and links with distances (COST239)
        nodes = [i for i in range(1, 12)]  # Use integers for nodes
        links = [
            (1, 2, 953), (1, 3, 622), (1, 4, 361), (1, 7, 641),
            (2, 3, 356), (2, 5, 321), (2, 8, 343),
            (3, 4, 576), (3, 5, 171), (3, 6, 318),
            (4, 7, 281), (4, 8, 877), (4, 10, 525),
            (5, 6, 190), (5, 8, 266), (5, 11, 697),
            (6, 7, 594), (6, 8, 294), (6, 9, 251),
            (7, 9, 529), (7, 10, 251),
            (8, 9, 490), (8, 11, 641),
            (9, 10, 594), (9, 11, 261), 
            (10, 11, 625)
        ]
    elif topology_name == "NSFNET":
        # Nodes and links with distances (NSFNET)
        nodes = [i for i in range(1, 15)]  # Use integers for nodes
        links = [
            (1, 4, 1136), (1, 8, 2828), (1, 11, 1702),
            (2, 3, 596), (2, 5, 2349), (2, 10, 789),
            (3, 9, 366), (3, 14, 385),
            (4, 5, 959), (4, 11, 683),
            (5, 6, 573), 
            (6, 7, 732), (6, 12, 1450),
            (7, 8, 750),
            (8, 9, 706), 
            (9, 10, 451), (9, 13, 839),
            (10, 14, 246), 
            (11, 12, 2049), 
            (12, 13, 1128), 
            (12, 14, 1976),
        ]
    elif topology_name == "DTGerman":
        # Define nodes as integers, mapping names to integers
        node_names = [
            "Norden", "Hamburg", "Bremen", "Berlin", "Hannover", "Essen", "Düsseldorf", "Köln",
            "Dortmund", "Leipzig", "Frankfurt", "Mannheim", "Nürnberg", "Karlsruhe", "Stuttgart", "Ulm", "München"
        ]
        node_mapping = {name: idx for idx, name in enumerate(node_names)}
        
        # Convert node names to integers
        nodes = [node_mapping[name] for name in node_names]
        
        # Update the links to use integer indices
        links = [
            (node_mapping["Norden"], node_mapping["Bremen"], 160),
            (node_mapping["Norden"], node_mapping["Dortmund"], 313),
            (node_mapping["Hamburg"], node_mapping["Bremen"], 124),
            (node_mapping["Hamburg"], node_mapping["Berlin"], 288),
            (node_mapping["Hamburg"], node_mapping["Hannover"], 159),
            (node_mapping["Bremen"], node_mapping["Hannover"], 132),
            (node_mapping["Berlin"], node_mapping["Hannover"], 291),
            (node_mapping["Berlin"], node_mapping["Leipzig"], 195),
            (node_mapping["Hannover"], node_mapping["Dortmund"], 183),
            (node_mapping["Hannover"], node_mapping["Leipzig"], 264),
            (node_mapping["Hannover"], node_mapping["Frankfurt"], 352),
            (node_mapping["Essen"], node_mapping["Düsseldorf"], 35),
            (node_mapping["Essen"], node_mapping["Dortmund"], 37),
            (node_mapping["Düsseldorf"], node_mapping["Köln"], 42),
            (node_mapping["Köln"], node_mapping["Dortmund"], 95),
            (node_mapping["Köln"], node_mapping["Frankfurt"], 194),
            (node_mapping["Leipzig"], node_mapping["Frankfurt"], 393),
            (node_mapping["Leipzig"], node_mapping["Nürnberg"], 281),
            (node_mapping["Frankfurt"], node_mapping["Mannheim"], 79),
            (node_mapping["Frankfurt"], node_mapping["Nürnberg"], 225),
            (node_mapping["Mannheim"], node_mapping["Karlsruhe"], 66),
            (node_mapping["Nürnberg"], node_mapping["Stuttgart"], 208),
            (node_mapping["Nürnberg"], node_mapping["München"], 166),
            (node_mapping["Karlsruhe"], node_mapping["Stuttgart"], 80),
            (node_mapping["Stuttgart"], node_mapping["Ulm"], 281),
            (node_mapping["Ulm"], node_mapping["München"], 156),
        ]
    elif topology_name == "UBN":
        # Nodes and links with distances (UBN)
        nodes = [i for i in range(1, 25)]  # Use integers for nodes
        links = [
            (1, 2, 800), (1, 6, 1000), 
            (2, 3, 1100), (2, 6, 950),
            (3, 4, 250), (3, 5, 1000), (3, 7, 1000),
            (4, 5, 850), (4, 7, 850),
            (5, 8, 1200),
            (6, 7, 1000), (6, 9, 1200), (6, 11, 1900),
            (7, 8, 1150), (7, 9, 1000), 
            (8, 10, 900),
            (9, 10, 1000), (9, 11, 1400), (9, 12, 1000),
            (10, 13, 950), (10, 14, 850),
            (11, 12, 900), (11, 15, 1300),
            (12, 13, 900), (12, 16, 1000),
            (13, 14, 650), (13, 17, 1100),
            (14, 18, 1200),
            (15, 16, 600), (15, 19, 2600), (15, 20, 1300),
            (16, 17, 1000), (16, 21, 1000), (16, 22, 800),
            (17, 18, 800), (17, 22, 850), (17, 23, 1000),
            (18, 24, 900),
            (19, 20, 959),
            (20, 21, 700),
            (21, 22, 300),
            (22, 23, 600),
            (23, 24, 900),
        ]   
    elif topology_name == "CONUS30":
        # Nodes and links with distances (CONUS30)
        nodes = [i for i in range(1, 31)]  # Use integers for nodes
        links = [
            (1, 5, 457), (1, 10, 1467), 
            (2, 19, 166), (2, 30, 72),
            (3, 6, 1190), (3, 18, 398),
            (4, 6, 721), (4, 11, 357),
            (5, 15, 1365), (5, 30, 730),
            (7, 10, 465), (7, 12, 1060), 
            (8, 9, 1160), (8, 12, 1157), (8, 22, 850),
            (9, 20, 720), (9, 23, 979),
            (10, 17, 686), (10, 23, 380),
            (11, 16, 680), (11, 26, 487),
            (12, 26, 490),
            (13, 20, 767), (13, 25, 626),
            (14, 20, 526), (14, 22, 773),
            (15, 28, 430),
            (16, 30, 415),
            (17, 29, 703), 
            (18, 19, 165),
            (21, 22, 1110), (21, 24, 180), (21, 27, 1374),
            (22, 27, 1463),
            (24, 25, 69),
            (28, 29, 426),
        ]   
    elif topology_name == "CONUS60":
        # Nodes and links with distances (CONUS60)
        nodes = [i for i in range(1, 61)]  # Use integers for nodes
        links = [
            (1, 8, 277), (1, 53, 234),
            (2, 16, 648), (2, 18, 437),
            (3, 7 ,266), (3, 10, 439), (3, 22, 554),
            (4, 37, 179), (4, 59, 67),
            (5, 21, 500), (5, 32, 147),
            (6, 30, 1468), (6, 51, 1293),
            (7, 31, 352), (7, 32, 583),
            (8, 41, 80),
            (9 ,13, 336), (9, 53, 272),
            (10, 19, 160),
            (11 , 17, 459), (11, 29, 165), (11, 52, 502),
            (12, 14, 193), (12, 27, 178), (12, 59, 777),
            (13, 14, 239), (13, 56, 191),
            (14, 39, 295),
            (15, 18, 1190), (15, 21, 433), (15, 25, 554), (15, 58, 560),
            (16, 36, 920), (16, 44, 731),
            (17, 56, 107),
            (18, 45, 965), (18, 57, 506), 
            (19, 27, 690), (19, 42, 131), (19, 59, 508),
            (20, 33, 216), (20, 41, 126),
            (21, 45, 425),
            (22, 42, 809), (22, 60, 522),
            (23, 36, 314), (23, 52, 471), (23, 58, 418),
            (24, 26, 485), (24, 35, 786), (24, 38, 496), (24, 44, 700),
            (25, 31, 639),
            (26, 46, 224), (26, 49, 151),
            (27, 31, 295), (27, 52, 474),
            (28, 55, 397), (28, 60, 130),
            (29, 30, 568),
            (30, 36, 561),
            (32, 54, 653),
            (33, 34, 24), (33, 50, 200),
            (34, 37, 136),
            (35, 43, 133), (35, 44, 1136), (35, 47, 26),
            (37, 50, 193),
            (38, 46, 575), (38, 57, 223),
            (39, 50, 474),
            (40, 43, 938), (40, 51, 279),
            (44, 51, 1463),
            (47, 48, 77),
            (48, 49, 447),
            (50, 53 , 224),
            (54, 55, 394),
        ]  
    elif topology_name == "PTbackbone":
        # Define nodes as integers, mapping names to integers
        node_names = [
            "Alcácer do Sal", "Aveiro", "Beja", "Braga", "Bragança", "Caldas da Rainha", "Castelo Branco", "Coimbra",
            "Elvas", "Évora", "Faro", "Funchal", "Guarda", "Leiria", "Lisboa", "Ponta Delgada", "Portalegre", "Portimão",
            "Porto", "Santarém", "São João da Madeira", "Setúbal", "Sines", "Viana do Castelo", "Vila Real", "Viseu"
        ]
        node_mapping = {name: idx for idx, name in enumerate(node_names)}
        
        # Convert node names to integers
        nodes = [node_mapping[name] for name in node_names]
        
        # Update the links to use integer indices
        links = [
            (node_mapping["Alcácer do Sal"], node_mapping["Évora"], 68),
            (node_mapping["Alcácer do Sal"], node_mapping["Setúbal"], 50),
            (node_mapping["Alcácer do Sal"], node_mapping["Sines"], 65),
            (node_mapping["Aveiro"], node_mapping["Coimbra"], 58),
            (node_mapping["Aveiro"], node_mapping["Leiria"], 113),
            (node_mapping["Aveiro"], node_mapping["Porto"], 67),
            (node_mapping["Beja"], node_mapping["Évora"], 76),
            (node_mapping["Beja"], node_mapping["Faro"], 140),
            (node_mapping["Braga"], node_mapping["Bragança"], 209),
            (node_mapping["Braga"], node_mapping["Porto"], 53),
            (node_mapping["Braga"], node_mapping["São João da Madeira"], 85),
            (node_mapping["Braga"], node_mapping["Viana do Castelo"], 47),
            (node_mapping["Bragança"], node_mapping["Vila Real"], 118),
            (node_mapping["Caldas da Rainha"], node_mapping["Leiria"], 54),
            (node_mapping["Caldas da Rainha"], node_mapping["Lisboa"], 88),
            (node_mapping["Castelo Branco"], node_mapping["Guarda"], 94),
            (node_mapping["Castelo Branco"], node_mapping["Portalegre"], 80),
            (node_mapping["Coimbra"], node_mapping["Santarém"], 136),
            (node_mapping["Coimbra"], node_mapping["São João da Madeira"], 84),
            (node_mapping["Coimbra"], node_mapping["Viseu"], 84),
            (node_mapping["Elvas"], node_mapping["Évora"], 83),
            (node_mapping["Elvas"], node_mapping["Portalegre"], 56),
            (node_mapping["Faro"], node_mapping["Portimão"], 62),
            (node_mapping["Funchal"], node_mapping["Lisboa"], 1050),
            (node_mapping["Funchal"], node_mapping["Ponta Delgada"], 1050),
            (node_mapping["Funchal"], node_mapping["Portimão"], 1010),
            (node_mapping["Guarda"], node_mapping["Viseu"], 73),
            (node_mapping["Leiria"], node_mapping["Santarém"], 70),
            (node_mapping["Lisboa"], node_mapping["Ponta Delgada"], 1500),
            (node_mapping["Lisboa"], node_mapping["Santarém"], 76),
            (node_mapping["Lisboa"], node_mapping["Setúbal"], 44),
            (node_mapping["Portalegre"], node_mapping["Santarém"], 144),
            (node_mapping["Portimão"], node_mapping["Sines"], 139),
            (node_mapping["Porto"], node_mapping["São João da Madeira"], 32),
            (node_mapping["Porto"], node_mapping["Viana do Castelo"], 70),
            (node_mapping["Vila Real"], node_mapping["Viseu"], 97),
        ]
    elif topology_name == "COST266":
        # # Define the node coordinates in degrees
        nodes = {
            "Amsterdam": (52.35, 4.90),
            "Athens": (38.00, 23.73),
            "Barcelona": (41.37, 2.18),
            "Belgrade": (44.83, 20.50),
            "Berlin": (52.52, 13.40),
            "Birmingham": (52.47, -1.88),
            "Bordeaux": (44.85, -0.57),
            "Brussels": (50.83, 4.35),
            "Budapest": (47.50, 19.08),
            "Copenhagen": (55.72, 12.57),
            "Dublin": (53.33, -6.25),
            "Dusseldorf": (51.23, 6.78),
            "Frankfurt": (50.10, 8.67),
            "Glasgow": (55.85, -4.25),
            "Hamburg": (53.55, 10.02),
            "Helsinki": (60.17, 24.97),
            "Krakow": (50.05, 19.95),
            "Lisbon": (38.73, -9.13),
            "London": (51.50, -0.17),
            "Lyon": (45.73, 4.83),
            "Madrid": (40.42, -3.72),
            "Marseille": (43.30, 5.37),
            "Milan": (45.47, 9.17),
            "Munich": (48.13, 11.57),
            "Oslo": (59.93, 10.75),
            "Palermo": (38.12, 13.35),
            "Paris": (48.87, 2.33),
            "Prague": (50.08, 14.43),
            "Rome": (41.88, 12.50),
            "Seville": (37.38, -5.98),
            "Sofia": (42.75, 23.33),
            "Stockholm": (59.33, 18.05),
            "Strasbourg": (48.58, 7.77),
            "Vienna": (48.22, 16.37),
            "Warsaw": (52.25, 21.00),
            "Zagreb": (45.83, 16.02),
            "Zurich": (47.38, 8.55)
        }

        # Define the links between the nodes
        links = [
            ('Amsterdam', 'Brussels'), ('Amsterdam', 'Glasgow'), ('Amsterdam', 'Hamburg'),
            ('Amsterdam', 'London'), ('Athens', 'Palermo'), ('Athens', 'Sofia'),
            ('Athens', 'Zagreb'), ('Barcelona', 'Madrid'), ('Barcelona', 'Marseille'),
            ('Barcelona', 'Seville'), ('Belgrade', 'Budapest'), ('Belgrade', 'Sofia'),
            ('Belgrade', 'Zagreb'), ('Berlin', 'Copenhagen'), ('Berlin', 'Hamburg'),
            ('Berlin', 'Munich'), ('Berlin', 'Prague'), ('Berlin', 'Warsaw'),
            ('Birmingham', 'Glasgow'), ('Birmingham', 'London'), ('Bordeaux', 'Madrid'),
            ('Bordeaux', 'Marseille'), ('Bordeaux', 'Paris'), ('Brussels', 'Dusseldorf'),
            ('Brussels', 'Paris'), ('Budapest', 'Krakow'), ('Budapest', 'Prague'),
            ('Copenhagen', 'Oslo'), ('Copenhagen', 'Stockholm'), ('Dublin', 'Glasgow'),
            ('Dublin', 'London'), ('Dusseldorf', 'Frankfurt'), ('Frankfurt', 'Hamburg'),
            ('Frankfurt', 'Munich'), ('Frankfurt', 'Strasbourg'), ('Helsinki', 'Oslo'),
            ('Helsinki', 'Stockholm'), ('Helsinki', 'Warsaw'), ('Krakow', 'Warsaw'),
            ('Lisbon', 'London'), ('Lisbon', 'Madrid'), ('Lisbon', 'Seville'),
            ('London', 'Paris'), ('Lyon', 'Marseille'), ('Lyon', 'Paris'),
            ('Lyon', 'Zurich'), ('Marseille', 'Rome'), ('Milan', 'Munich'),
            ('Milan', 'Rome'), ('Milan', 'Zurich'), ('Munich', 'Vienna'),
            ('Palermo', 'Rome'), ('Paris', 'Strasbourg'), ('Prague', 'Vienna'),
            ('Rome', 'Zagreb'), ('Strasbourg', 'Zurich'), ('Vienna', 'Zagreb')
        ]

        # Convert the coordinates to radians
        node_coords_rad = np.radians(list(nodes.values()))

        # Create a dictionary to map node names to indices
        node_indices = {node: idx for idx, node in enumerate(nodes)}

        # Calculate the haversine distance matrix
        dist_matrix = haversine_distances(node_coords_rad) * 6371  # Multiply by Earth's radius to get distance in km

        # Initialize the graph
        G = nx.Graph()

        # Add nodes with positions
        for node, (lat, lon) in nodes.items():
            idx = node_indices[node]  # Get the integer index
            G.add_node(idx, pos=(lon, lat))

        # Add edges with weights based on the distance matrix
        for (node1, node2) in links:
            idx1 = node_indices[node1]  # Convert node1 to its integer index
            idx2 = node_indices[node2]  # Convert node2 to its integer index

            lat1, lon1 = nodes[node1]
            lat2, lon2 = nodes[node2]
            coord1 = np.radians([lat1, lon1])
            coord2 = np.radians([lat2, lon2])
            distance = haversine_distances([coord1, coord2])[0][1] * 6371  # Distance in km
            distance = round(distance, 2)
            G.add_edge(idx1, idx2, weight=distance)

        return G
    elif topology_name.startswith("ring"):
        # Generate ring networks for different node counts
        node_count = int(topology_name.split("-")[1])  # Example input: "ring-50"
        G = create_ring_network(node_count)
        return G
    else:
        raise ValueError("Unknown topology name.")

    # Add nodes and edges to the graph
    G.add_nodes_from(nodes)
    for link in links:
        G.add_edge(link[0], link[1], weight=link[2])

    return G

# Function to calculate and display network characteristics
def calculate_topological_characteristics(G):
    if verbose:
        print("----- NETWORK TOPOLOGICAL CHARACTERISTICS -----")
        link_lengths = [d['weight'] for (u, v, d) in G.edges(data=True)]
        node_degrees = [deg for _, deg in G.degree()]
        print("1 - Number of Nodes:", G.number_of_nodes())
        print("2 - Number of Links:", G.number_of_edges())
        print("3 - Minimum Link Length (km):", min(link_lengths))
        print("4 - Maximum Link Length (km):", max(link_lengths))
        print("5 - Average Link Length (km):", np.mean(link_lengths))
        print("6 - Variance of Link Length (km^2):", np.var(link_lengths))
        print("7 - Minimum Node Degree:", min(node_degrees))
        print("8 - Maximum Node Degree:", max(node_degrees))
        print("9 - Average Node Degree:", np.mean(node_degrees))
        print("10 - Variance of Node Degree:", np.var(node_degrees))
        print("11 - Network Diameter:", nx.diameter(G, weight="weight"))
        print("12 - Algebraic Connectivity:", nx.algebraic_connectivity(G, weight="weight"))

# Additional function to calculate and display routing results and link loads
def display_routing_results(G, available_wavelengths, successful_connections, blocked_connections, Cnet, num_channels):
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

# Helper function to determine the number of required wavelengths
def calculate_required_wavelengths(available_wavelengths):
    """
    Calculate the maximum number of wavelengths used across all links.

    Parameters:
    - available_wavelengths: Dictionary indicating wavelength availability on each link.

    Returns:
    - required_wavelengths: The maximum number of used wavelengths.
    """
    max_wavelength_used = 0
    for link, wavelengths in available_wavelengths.items():
        # Count the highest used wavelength (first False from the end) for this link
        highest_used = len(wavelengths) - 1 - wavelengths[::-1].index(False) if False in wavelengths else 0
        max_wavelength_used = max(max_wavelength_used, highest_used + 1)  # +1 because wavelengths are 0-indexed
    return max_wavelength_used

# -------------------------------------------------------------------------------- #
# --------------------------- EXPLORING RWA RESULTS ------------------------------ #
# -------------------------------------------------------------------------------- #

# Flags
verbose = True  # Enable detailed output

# Simulation parameters
path_order_strategy = "longest-first"         # Options: "shortest-first", "longest-first", "random"
wavelength_assignment_strategy = "first-fit"   # Options: "first-fit", "most-used", "least-used", "random"
trials = 1                                     # Number of simulation runs per network
k_paths = 2                                    # Number of k-shortest paths to consider

# Define WDM channel capacity
num_channels_Super_CL = 1500  

# List to store average wavelengths used for each network
wavelengths_used = []

# Define the networks to simulate
network_list = ["COST239"]  # Add more networks if needed
# network_list = ["CESNET", "COST239", "NSFNET", "DTGerman", "UBN", "PTbackbone", "CONUS30", "COST266", "CONUS60", "ring-25", "ring-35", "ring-45"]

for topology_name in network_list:
    print(f"\nAnalyzing {topology_name} Network over {trials} trials")
    network_wavelengths_used = []  # Stores required wavelengths per trial
    
    # Start simulation for each network
    start_time = time.perf_counter()

    for trial in range(trials):
        print(f"\n--- Trial {trial + 1} of {trials} ---")

        # Load the network graph, calculate characteristics, and visualize (optional)
        G = define_network(topology_name)
        # Define network graph, calculate characteristics, and visualize (optional)
        calculate_topological_characteristics(G)
        # visualize_graph(G, title=f"{topology_name} Network Topology")

        # ------------------------------------- #
        # STEP 1: Define Traffic Matrix         #
        # ------------------------------------- #
        nodes = list(G.nodes)

        # Ensure nodes are integers for sorting consistency
        try:
            nodes = sorted(nodes, key=int)
        except ValueError:
            raise ValueError("Graph nodes must be convertible to integers for sorting.")

        # Default full-mesh: 1 demand per pair (source < target)
        traffic_matrix = {(i, j): 1 for i in nodes for j in nodes if int(i) < int(j)}

        # ------------------------------------- #
        # STEP 2: Compute k-Shortest Paths      #
        # ------------------------------------- #
        # all_pairs_shortest_paths = compute_paths_for_traffic_matrix(G, traffic_matrix, k=k_paths, weight='weight')
        all_pairs_shortest_paths = compute_paths_for_traffic_matrix_with_tie_breaking(G, traffic_matrix, k=k_paths, weight='weight')

        # ------------------------------------- #
        # STEP 3: Order Traffic Demands         #
        # ------------------------------------- #
        sorted_traffic_demands = order_traffic_demands(G, all_pairs_shortest_paths, traffic_matrix, path_order_strategy)
        for sorted_demand in sorted_traffic_demands:
            for path in sorted_demand[2]:
                print(f'{sorted_demand[0]} -> {sorted_demand[1]}: Length = {path_distance(G, path)}: Path {path}')

        # ------------------------------------- #
        # STEP 4: Wavelength Assignment         #
        # ------------------------------------- #
        # Initialize wavelength availability for each link
        available_wavelengths = {(u, v): [True] * num_channels_Super_CL for u, v in G.edges}
        available_wavelengths.update({(v, u): [True] * num_channels_Super_CL for u, v in G.edges})

        # Assign wavelengths and track blocked connections
        successful_connections, blocked_connections = handle_traffic_demands(
            sorted_traffic_demands, 
            available_wavelengths, 
            num_channels_Super_CL, 
            k=k_paths, 
            wavelength_assignment_strategy=wavelength_assignment_strategy
        )

        # ------------------------------------- #
        # STEP 5: Compute Required Wavelengths #
        # ------------------------------------- #
        required_wavelengths = calculate_required_wavelengths(available_wavelengths)
        network_wavelengths_used.append(required_wavelengths)
        
        print(f"Required wavelengths in Trial {trial + 1}: {required_wavelengths}")

        # Optional: Display routing results
        # display_routing_results(G, available_wavelengths, successful_connections, blocked_connections, 0, num_channels_Super_CL)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Total Execution time for {topology_name}: {execution_time:.2f} seconds")

    # ------------------------------------- #
    # STEP 6: Compute Average Wavelengths Used #
    # ------------------------------------- #
    avg_wavelengths = np.mean(network_wavelengths_used)
    wavelengths_used.append(avg_wavelengths)
    
    print(f"\nAverage number of required wavelengths for {topology_name} over {trials} trials: {avg_wavelengths}")
    print(f"Required wavelengths for {topology_name} with {path_order_strategy} ordering and {wavelength_assignment_strategy} assignment: {required_wavelengths}")

# ------------------------------------- #
# STEP 7: Print Final Report           #
# ------------------------------------- #
for idx, network in enumerate(network_list):
    print(f"{network}: Average Required Wavelengths over {trials} trials = {wavelengths_used[idx]}")

# ----------------------------------------------------------------------- #
# ------------------------- EXPLORE k-SHORTEST PATHS -------------------- #
# ----------------------------------------------------------------------- #

# Simulation parameters
path_order_strategy = "longest-first"  # Options: "shortest-first", "longest-first", "random"
wavelength_assignment_strategy = "first-fit"  # Options: "first-fit", "most-used", "least-used", "random"
k_values = list(range(1, 21))  # Range of k values to test

# Networks to analyze
network_topologies = ["COST239"]  # Modify this list to include other networks
minimum_wavelengths = {network: [] for network in network_topologies}

# Iterate over each network topology
for topology_name in network_topologies:
    print(f"\nAnalyzing {topology_name} Network")
    
    # Define network graph
    G = define_network(topology_name)
    
    for k_paths in k_values:
        print(f"\n--- Testing k = {k_paths} ---")
        
        # Store required wavelengths for this k value
        required_wavelengths_for_k = []
        start_time = time.perf_counter()

        # ------------------------------------- #
        # STEP 1: Define Traffic Matrix         #
        # ------------------------------------- #
        nodes = list(G.nodes)

        # Ensure nodes are integers for sorting consistency
        try:
            nodes = sorted(nodes, key=int)
        except ValueError:
            raise ValueError("Graph nodes must be convertible to integers for sorting.")

        # Default full-mesh: 1 demand per pair (source < target)
        traffic_matrix = {(i, j): 1 for i in nodes for j in nodes if int(i) < int(j)}

        # ------------------------------------- #
        # STEP 2: Compute k-Shortest Paths      #
        # ------------------------------------- #
        all_pairs_shortest_paths = compute_paths_for_traffic_matrix(G, traffic_matrix, k=k_paths, weight='weight')

        # ------------------------------------- #
        # STEP 3: Order Traffic Demands         #
        # ------------------------------------- #
        sorted_traffic_demands = order_traffic_demands(G, all_pairs_shortest_paths, traffic_matrix, path_order_strategy)

        # ---------- Step 4 + 5: Wavelength Assignment and Blocking ---------- #
        # Set the initial number of wavelengths based on network topology
        initial_wavelengths = {
            "CESNET": 1, "COST239": 1, "NSFNET": 10, "UBN": 30, "CONUS30": 100,
            "PTbackbone": 70, "COST266": 100, "CONUS60": 490, "ring-25": 70, 
            "ring-35": 140, "ring-45": 230
        }
        num_wavelengths = initial_wavelengths.get(topology_name, 1)  # Default to 1 if topology not in dictionary

        while True:
            print(f"Testing with {num_wavelengths} wavelengths available")

            # Initialize wavelength availability for each link in both directions
            available_wavelengths = {(u, v): [True] * num_wavelengths for u, v in G.edges}
            available_wavelengths.update({(v, u): [True] * num_wavelengths for u, v in G.edges})

            # Assign wavelengths to paths and handle blocked connections
            successful_connections, blocked_connections = handle_traffic_demands(
                sorted_traffic_demands, 
                available_wavelengths, 
                num_wavelengths, 
                k=k_paths, 
                wavelength_assignment_strategy=wavelength_assignment_strategy
            )

            # Check for blocked demands
            if not blocked_connections:
                print(f"No blocking with {num_wavelengths} wavelengths")
                minimum_wavelengths[topology_name].append(num_wavelengths)
                break
            else:
                num_wavelengths += 1

# ---------- Step 5: Plot Results ---------- #
plt.figure(figsize=(8, 5))
for topology_name, wavelengths in minimum_wavelengths.items():
    plt.plot(k_values, wavelengths, marker='o', label=topology_name)

# Set x-axis and y-axis ticks
plt.xticks(k_values)
y_min = min(min(wavelengths) for wavelengths in minimum_wavelengths.values()) - 5
y_max = max(max(wavelengths) for wavelengths in minimum_wavelengths.values()) + 5
plt.yticks(range(y_min, y_max, 5))  # Y-axis step of 5

# Plot labels and title
plt.xlabel("k (Number of k-Shortest Paths Considered)")
plt.ylabel("Number of Required Wavelengths")
plt.title("Required Wavelengths as a Function of k for Various Networks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------- #
# ---------------------- EXPLORE LINK LOAD DISTRIBUTION ----------------- #
# ----------------------------------------------------------------------- # 

# Simulation parameters
path_order_strategies = ["shortest-first", "longest-first", "random"]
k_paths = 2  # Define the k value to evaluate
num_channels_Super_CL = 6  # Define the number of channels

topology_name = "CESNET"

# Store results for each strategy
link_wavelength_usage = {strategy: {} for strategy in path_order_strategies}
wavelengths_used = []

# Iterate over each path ordering strategy
for path_order_strategy in path_order_strategies:
    print(f"\nAnalyzing {topology_name} Network using {path_order_strategy} ordering")
    
    # ------------------------------------- #
    # STEP 1: Define Network Graph         #
    # ------------------------------------- #
    G = define_network(topology_name)
    
    # ------------------------------------- #
    # STEP 2: Define Traffic Matrix        #
    # ------------------------------------- #
    nodes = list(G.nodes)
    try:
        nodes = sorted(nodes, key=int)
    except ValueError:
        raise ValueError("Graph nodes must be convertible to integers for sorting.")
    
    traffic_matrix = {(i, j): 1 for i in nodes for j in nodes if int(i) < int(j)}
    
    # ------------------------------------- #
    # STEP 3: Compute k-Shortest Paths     #
    # ------------------------------------- #
    all_pairs_shortest_paths = compute_paths_for_traffic_matrix(G, traffic_matrix, k=k_paths, weight='weight')
    
    # ------------------------------------- #
    # STEP 4: Order Traffic Demands        #
    # ------------------------------------- #
    sorted_traffic_demands = order_traffic_demands(G, all_pairs_shortest_paths, traffic_matrix, path_order_strategy)
    
    for sorted_demand in sorted_traffic_demands:
        for path in sorted_demand[2]:
            print(f'{sorted_demand[0]} -> {sorted_demand[1]}: Length = {sorted_demand[3]}: Path {path}')
    
    # ------------------------------------- #
    # STEP 5: Wavelength Assignment        #
    # ------------------------------------- #
    available_wavelengths = {(u, v): [True] * num_channels_Super_CL for u, v in G.edges}
    available_wavelengths.update({(v, u): [True] * num_channels_Super_CL for u, v in G.edges})
    
    successful_connections, blocked_connections = handle_traffic_demands(
        sorted_traffic_demands,
        available_wavelengths,
        num_channels_Super_CL,
        k=k_paths,
        wavelength_assignment_strategy="first-fit"
    )
    
    # Calculate the number of wavelengths used per link
    for (u, v), wavelengths in available_wavelengths.items():
        used_count = num_channels_Super_CL - sum(wavelengths)
        if (v, u) not in link_wavelength_usage[path_order_strategy]:
            link_wavelength_usage[path_order_strategy][(u, v)] = used_count
    
    # Calculate the required wavelengths
    required_wavelengths = calculate_required_wavelengths(available_wavelengths)
    wavelengths_used.append(required_wavelengths)
    print(f"\n{path_order_strategy} strategy required wavelengths: {required_wavelengths}")
    
    display_routing_results(G, available_wavelengths, successful_connections, blocked_connections, 0, num_channels_Super_CL)

# ------------------------------------- #
# STEP 6: Plot Wavelength Usage        #
# ------------------------------------- #
fig, ax = plt.subplots()
colors = {"shortest-first": "blue", "longest-first": "green", "random": "orange"}
bar_width = 0.15
unique_links = sorted(set(min(link, (link[1], link[0])) for strategy in link_wavelength_usage for link in link_wavelength_usage[strategy]))
x = np.arange(len(unique_links))

for i, strategy in enumerate(path_order_strategies):
    link_data = link_wavelength_usage[strategy]
    y_values = [link_data.get(link, 0) for link in unique_links]
    ax.bar(x + i * bar_width, y_values, bar_width, label=strategy.capitalize(), color=colors[strategy])

ax.set_xlabel("Links")
ax.set_ylabel("Number of Wavelengths Used")
ax.set_xticks(x + bar_width)
ax.set_xticklabels([f"{u}-{v}" for (u, v) in unique_links], rotation=90)
ax.legend()
ax.grid(True, axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()

# # ----------------------------------------------------------------------- #
# # -------------------- EXPLORE (UN)CONSTRAINED ROUTING  ----------------- #
# # ----------------------------------------------------------------------- # 

# # topologies = ["CESNET", "COST239", "UBN"]
# topologies = ["CESNET"]
# path_order_strategies = ["shortest-first", "longest-first", "random"]
# k_values = [2]  # Define the k values you want to evaluate
# infinite_channels = 10000  # "Infinite" number of channels for unconstrained routing
# constrained_channels = 6  # Defined limit for constrained routing
# # Results storage
# wavelengths_used = {topology: [] for topology in topologies}
# link_wavelength_usage_constrained = {
#     topology: {strategy: {k: {} for k in k_values} for strategy in path_order_strategies} for topology in topologies
# }
# link_wavelength_usage_unconstrained = {
#     topology: {strategy: {k: {} for k in k_values} for strategy in path_order_strategies} for topology in topologies
# }

# # Loop over each topology
# for topology_name in topologies:
#     print(f"\nAnalyzing {topology_name} Network for RWA results")
#     G = define_network(topology_name)

#     for path_order_strategy in path_order_strategies:
#         for k_paths in k_values:
#             print(f"\nTopology: {topology_name} | Strategy: {path_order_strategy} | k = {k_paths}")
#             all_pairs_shortest_paths = {}

#             # ---------- Step 1: Paths Computation ---------- #
#             for source in G.nodes:
#                 for target in G.nodes:
#                     if source < target:  # Compute paths only in one direction
#                         paths = k_shortest_paths(G, source, target, k=k_paths, weight='weight')
#                         all_pairs_shortest_paths[(source, target)] = paths

#             # ---------- Step 2: Order the Traffic Demands ---------- #
#             sorted_traffic_demands = order_traffic_demands(all_pairs_shortest_paths, path_order_strategy)

#             # Run simulations for both constrained and unconstrained routing
#             for routing_type, num_channels in [("Unconstrained", infinite_channels), ("Constrained", constrained_channels)]:
#                 print(f"\nRunning {routing_type} Routing for {topology_name} with {num_channels} channels")

#                 # Initialize wavelength availability for each link in both directions
#                 available_wavelengths = {(u, v): [True] * int(num_channels) for u, v in G.edges}
#                 available_wavelengths.update({(v, u): [True] * int(num_channels) for u, v in G.edges})

#                 # Assign wavelengths to paths and handle blocked connections
#                 successful_connections, blocked_connections = handle_traffic_demands(
#                     sorted_traffic_demands,
#                     available_wavelengths,
#                     int(num_channels),
#                     k=k_paths,
#                     wavelength_assignment_strategy="first-fit"
#                 )

#                 # Calculate the number of wavelengths used for each link
#                 link_usage = {}
#                 for (u, v), wavelengths in available_wavelengths.items():
#                     used_count = len(wavelengths) - sum(wavelengths)  # Count used wavelengths
#                     if (v, u) not in link_usage:  # Avoid duplicates
#                         link_usage[(u, v)] = used_count

#                 # Store results for constrained and unconstrained routing
#                 if routing_type == "Unconstrained":
#                     link_wavelength_usage_unconstrained[topology_name][path_order_strategy][k_paths] = link_usage
#                 else:
#                     link_wavelength_usage_constrained[topology_name][path_order_strategy][k_paths] = link_usage

# # ---------- Plot Results ---------- #

# # Set up subplots
# fig, axes = plt.subplots(len(topologies), 2, figsize=(14, 5 * len(topologies)), sharey=True)
# fig.suptitle("Link Wavelength Usage for Constrained vs Unconstrained Routing")

# # Check if there's only one topology to avoid indexing issues
# if len(topologies) == 1:
#     axes = np.array([axes])  # Wrap axes in an array to handle indexing

# # Plot results for each topology and strategy
# for i, topology_name in enumerate(topologies):
#     unique_links = sorted(set(link for strategy in path_order_strategies 
#                               for k in k_values 
#                               for link in link_wavelength_usage_constrained[topology_name][strategy][k]))

#     # Plot Constrained Routing
#     axes[i, 0].set_title(f"{topology_name} - Constrained Routing")
#     for j, strategy in enumerate(path_order_strategies):
#         for k_paths in k_values:
#             link_data = link_wavelength_usage_constrained[topology_name][strategy][k_paths]
#             y_values = [link_data.get(link, 0) for link in unique_links]
#             offset = j * 0.25 + (k_paths - 1) * 0.05  # Adjust for each strategy and k-value
#             axes[i, 0].bar(np.arange(len(unique_links)) + offset, y_values, 0.2, 
#                            label=f"{strategy.capitalize()} (k={k_paths})")
#     axes[i, 0].set_ylabel("Wavelengths Used")
#     axes[i, 0].set_xlabel("Links")
#     axes[i, 0].set_xticks(np.arange(len(unique_links)))
#     axes[i, 0].set_xticklabels([f"{u}-{v}" for (u, v) in unique_links], rotation=90)
#     axes[i, 0].legend()
#     axes[i, 0].grid(True, axis="y", linestyle="--", alpha=0.7)

#     # Plot Unconstrained Routing
#     axes[i, 1].set_title(f"{topology_name} - Unconstrained Routing")
#     for j, strategy in enumerate(path_order_strategies):
#         for k_paths in k_values:
#             link_data = link_wavelength_usage_unconstrained[topology_name][strategy][k_paths]
#             y_values = [link_data.get(link, 0) for link in unique_links]
#             offset = j * 0.25 + (k_paths - 1) * 0.05
#             axes[i, 1].bar(np.arange(len(unique_links)) + offset, y_values, 0.2, 
#                            label=f"{strategy.capitalize()} (k={k_paths})")
#     axes[i, 1].set_xlabel("Links")
#     axes[i, 1].set_xticks(np.arange(len(unique_links)))
#     axes[i, 1].set_xticklabels([f"{u}-{v}" for (u, v) in unique_links], rotation=90)
#     axes[i, 1].legend()
#     axes[i, 1].grid(True, axis="y", linestyle="--", alpha=0.7)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()