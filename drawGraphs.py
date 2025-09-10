import networkx as nx
import matplotlib.pyplot as plt
import os

def draw_graph_from_file(folder_path, graph_name=None):
    """
    Draws a graph or all graphs from a specified folder.
    If a graph_name is provided, only that specific graph is drawn.
    """
    
    # Get all .graphml files in the folder
    graph_files = [f for f in os.listdir(folder_path) if f.endswith(".graphml")]
    
    if not graph_files:
        print("No graph files found in the specified folder.")
        return
    
    if graph_name:
        # If a specific graph is requested, check if it exists
        file_name = f"{graph_name}.graphml"
        if file_name not in graph_files:
            print(f"Graph '{graph_name}' not found in the folder.")
            return
        graph_files = [file_name]
    
    for file in graph_files:
        file_path = os.path.join(folder_path, file)
        G = nx.read_graphml(file_path)
        
        # Extract node positions (stored as strings in GraphML, so they need conversion)
        pos = {}
        for node, data in G.nodes(data=True):
            if 'pos' in data:
                x, y = map(float, data['pos'].split(','))
                pos[node] = (x, y)
        
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, node_size=150, font_size=8, node_color='lightblue', edge_color='gray')
        edge_weights = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8)
        plt.title(f"Graph: {file}")
        plt.axis('off')
        plt.show()

# Example usage:
# draw_graph_from_file("graphs_extra")  # Draw all graphs in the folder
draw_graph_from_file("graphs_extra", "4_8-neighbors60_120_sim3")  # Draw a specific graph
