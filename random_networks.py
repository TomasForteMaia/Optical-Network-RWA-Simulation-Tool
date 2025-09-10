# Developed by Filipe Carmo 93054 MEEC

import networkx as nx
import random
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import os

# Average node degree
def avgNodeDegree(G, N, L):
    avg_node_degree = (2*L) / N
    
    return avg_node_degree

# Computes the degree of each node
def nodeDegree(G):
    nodeDegreeList = [G.degree(node) for node in G.nodes()]
    
    return nodeDegreeList

# Randomly place nodes in layout ensuring a minimum distance between nodes
def placeNodes(N, min_dist):
    pos = {}
    nodes = []
    
    # Layout dimension
    dim = int((9000 * 9000) ** 0.5)
    
    while len(pos) < N:
        p = (random.randint(0, dim), random.randint(0, dim))
        
        if pos:
            dist = min(distance.euclidean(p, q) for q in pos.values())
            if dist < min_dist:
                continue
        
        node = len(nodes)
        pos[node] = p
        nodes.append(node)
        
    return nodes, pos

# Returns the distance matrix with the distance between each pair of nodes
def distanceMatrix(G, N, pos):
    distList = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                dist = int(distance.euclidean(pos[source], pos[target]))
                distList.append(dist)      
            else:
                distList.append(0)
    # Distance matrix
    D = np.array(distList).reshape(N, N)
    
    return D

# Add edges to r1 neighbors
def r1Connect(G, D, r1, N):
    for source in G.nodes():
        neighbors = np.argsort(D[source])       # Sorts the neighbors in terms of euclidean distance
        neighbors = neighbors[1:]               # Exclude the node itself
        neighbors = neighbors[:r1]              # Pick the r1 closest neighbors
    
        for node in neighbors:
            G.add_edge(source, node, weight=D[source, node])
            
# Adds edges to nodes with degree 1 by choosing r2 random neighbors
def addEdgesDeg1(G, D, r2, degree1List, N):

    # Add an edge to all nodes with degree 1
    for source in degree1List:     
        if G.degree(source) != 1:
            continue
        else:
            neighbors = np.argsort(D[source])          # Sorts the neighbors in terms of euclidean distance
            neighbors = neighbors[1:]                  # Exclude the node itself
            r2List = neighbors[:r2]                    # Pick the r2 closest neighbors
            
            # Filter out neighbors that are already connected to the source
            r2List = [neighbor for neighbor in r2List if not G.has_edge(source, neighbor)]
            
            # Chooses a neighbor randomly
            random_neighbor = random.choice(r2List)   
            G.add_edge(source, random_neighbor, weight = D[source, random_neighbor])
            
    return G

# Ensures edge connectivity is 2
def edge_connectivity2(G, D):
    C = nx.complement(G)
    edges = C.edges()
    comp_edges = [(s, t, {'weight': D[s, t]}) for s, t in edges]
    complement = list(nx.k_edge_augmentation(G, k=2, avail=comp_edges, partial=True, weight='weight'))
    for edge in complement:
        G.add_edge(edge[0], edge[1], weight=D[edge[0], edge[1]])
    
    return G
        
# Ensures node connectivity is 2
def biconnected_graph(G, D):
    bi_components = list(nx.biconnected_components(G))
    
    cut_nodes = list(nx.all_node_cuts(G, k=1))

    for i in range(len(bi_components)):
        for j in range(len(cut_nodes)):
            bi_components[i].difference_update(cut_nodes[j])
    
    bi_components = [s for s in bi_components if s]     # removes empty sets
    
    # Add nodes to the graph
    S = nx.Graph()
    for node_set in bi_components:
        S.add_nodes_from(node_set)
    
    S = nx.complete_graph(S.nodes())
    
    # Add weights to edges
    for edge in S.edges():
        S[edge[0]][edge[1]]['weight'] = D[edge[0], edge[1]]
        
    # Define a minimum spanning tree
    S = nx.minimum_spanning_tree(S, weight='weight', algorithm='kruskal')
    
    G.add_edges_from(S.edges(data=True))
                
    return G

# Add random edges until max average node degree is met, and save each generated graph                   
def randomEdges(G, N, D, r2, min_avg_deg, max_avg_deg):
    
    graphList = []
    nodesList = list(G.nodes())
    
    if avgNodeDegree(G, N, G.number_of_edges()) >= min_avg_deg:
        if avgNodeDegree(G, N, G.number_of_edges()) <= max_avg_deg:
            graphList.append(G.copy())
        else:
            return graphList

    while avgNodeDegree(G, N, G.number_of_edges()) < max_avg_deg:
        random.shuffle(nodesList)
        for source in nodesList:
            neighbors = np.argsort(D[source])           # Sorts the neighbors in terms of euclidean distance
            neighbors = neighbors[1:]                   # Exclude the node itself
            r2List = neighbors[:r2]
            
            # Filter out neighbors that are already connected to the source
            r2List = [neighbor for neighbor in r2List if not G.has_edge(source, neighbor)]
            if r2List:
                random_neighbor = random.choice(r2List)
                G.add_edge(source, random_neighbor, weight=D[source, random_neighbor])
                
                # Save graph if it satifies minimum average node degree                        
                if avgNodeDegree(G, N, G.number_of_edges()) >= min_avg_deg:    
                    graphList.append(G.copy())
                    
                # Halt if maximum average node degree is met
                if avgNodeDegree(G, N, G.number_of_edges()) >= max_avg_deg:
                    break
                
    return graphList
                 
# Save graph in .graphml format and save graph plot
def saveGraph(graphList, pos, r1, r2, sim):
    
    for G in graphList:
    
        N = G.number_of_nodes()
        L = G.number_of_edges()
        
        # Set node position attributes 
        nx.set_node_attributes(G, pos, 'pos')
        # Convert positions to strings (GraphML writer does not support <class 'tuple'> as data values)
        positions = {n: f"{pos[0]},{pos[1]}" for n, pos in G.nodes(data='pos')}
        nx.set_node_attributes(G, positions, 'pos')
        
        # Define the graph name
        if N < 10:
            name = f'{r1}_{r2}-neighbors0{N}_{L}_sim{sim}'
        else:
            name = f'{r1}_{r2}-neighbors{N}_{L}_sim{sim}'
                
        # Write the graph to a .graphml file
        file_path = f'graphs_30_9000_9000/{name}.graphml'
        folder_path = 'graphs_30_9000_9000'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        nx.write_graphml(G, file_path)
        
        # Save graph drawing
        if DRAW:
            plt.figure(figsize=(10, 10))
            nx.draw_networkx(G, pos)
            edge_weights = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
            plt.axis('off')
            
            file_path = f'graphs plots/{name}.png'
            folder_path = 'graphs plots'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.savefig(file_path)
            plt.close()

#---------------------------------->> MAIN <<----------------------------------

# if True, draws the graph plot
DRAW = False

if __name__ == "__main__":
    
    # Network Parameters predefined:
        
    r1 = 4                  # initial number of Euclidian neighbors
    r2 = 8                  # number of Euclidian neighbors
    min_avg_deg = 2         # minimum average node degree
    max_avg_deg = 4.8       # maximum average node degree
    min_dist = 40           # minimum distance between nodes [km]
    simulations = 3         # number of simulations
    numNodesList = list(range(30, 31, 1))
    
    for N in numNodesList:
    
        for i in range(0, simulations):
            connected = False  # Initialize a flag for connectivity
            
            # Redo simulation if graph is not connected
            while not connected:
                    
                # Create Graph
                G = nx.Graph()
                
                # Place nodes in layout
                nodes, pos = placeNodes(N, min_dist)
                
                # Add nodes to the graph
                G.add_nodes_from(nodes)
                
                # Distance matrix
                D = distanceMatrix(G, N, pos)
        
                # Add edges to r1 neighbors
                r1Connect(G, D, r1, N)
                
                # Check if graph is connected
                if nx.is_connected(G):
                    connected = True        # If connected, exit while loop
            
            # Define a minimum spanning tree
            G = nx.minimum_spanning_tree(G, weight='weight', algorithm='kruskal')
                   
            # Compute degree of each node
            nodeDegreeList = nodeDegree(G)
            
            # Extracts all nodes with degree one
            degree1List = [index for index, degree in enumerate(nodeDegreeList) if degree == 1]
                
            # Adds an edge to all nodes with degree 1
            G = addEdgesDeg1(G, D, r2, degree1List, N)
                    
            # Ensures edge connectivity is 2
            if nx.edge_connectivity(G) < 2:
                G = edge_connectivity2(G, D)
                                                          
            # Ensures node connectivity is 2
            if nx.node_connectivity(G) < 2:
                G = biconnected_graph(G, D)
                
            # Add random edges until max avg deg is reached
            graphList = randomEdges(G, N, D, r2, min_avg_deg, max_avg_deg)
            
            if graphList:       
                # Save graph in a .graphml
                saveGraph(graphList, pos, r1, r2, i+1)
                
            print(f'Simulation {i+1} Complete.')