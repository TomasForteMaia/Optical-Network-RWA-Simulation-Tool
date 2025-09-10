import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import matplotlib.pyplot as plt
import os
import topohub
import json

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
            G.add_node(idx, pos=f"{lat},{lon}")

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

def save_graph(G, folder_name, graph_name, visualize=True):
    """
    Saves a graph to a specified folder in .graphml format and optionally saves its visualization.

    Parameters:
    - G: The NetworkX graph to save.
    - folder_name: The folder where the graph should be saved.
    - graph_name: The name of the graph file (without extension).
    - visualize: Whether to save a plot of the graph.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Convert node and edge attributes to GraphML-compatible formats
    for node, attrs in G.nodes(data=True):
        for key, value in attrs.items():
            if isinstance(value, (dict, list)):
                G.nodes[node][key] = json.dumps(value)

    for u, v, attrs in G.edges(data=True):
        for key, value in attrs.items():
            if isinstance(value, (dict, list)):
                G.edges[u, v][key] = json.dumps(value)

    # Convert graph-level attributes (like stats) to JSON strings
    for key, value in G.graph.items():
        if isinstance(value, (dict, list)):
            G.graph[key] = json.dumps(value)

    # Save the graph as a .graphml file
    graphml_path = os.path.join(folder_name, f"{graph_name}.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"Graph saved as: {graphml_path}")

    # Optionally visualize and save the graph plot
    if visualize:
        plt.figure(figsize=(10, 10))
        
        # Check if node positions exist
        pos = nx.get_node_attributes(G, 'pos')
 
        if pos:
            pos = {
                node: tuple(map(float, p.strip('[]').split(',')))
                for node, p in pos.items()
            }
        else:
            # Use spring layout if no positions are available
            pos = nx.spring_layout(G, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color="gray")
        
        # Use 'weight' as edge labels if it exists
        edge_labels = nx.get_edge_attributes(G, 'weight')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
        
        # Save the plot
        plot_folder = os.path.join(folder_name, "plots")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        plot_path = os.path.join(plot_folder, f"{graph_name}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Graph visualization saved as: {plot_path}")

# Save real networks
real_networks_folder = "real_networks"
real_topologies = ["CESNET", "COST239", "NSFNET", "DTGerman", "UBN", "PTbackbone", "CONUS30", "COST266", "CONUS60"]

for topology_name in real_topologies:
    G = define_network(topology_name)
    save_graph(G, real_networks_folder, topology_name)

# Save the Chinanet topology from TopoHub
topohub_networks_folder = "topohub_networks"
topo = topohub.get('topozoo/Chinanet')
G = nx.node_link_graph(topo)
save_graph(G, topohub_networks_folder, "Chinanet")