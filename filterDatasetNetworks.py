import os
import random
import shutil
import networkx as nx

def select_random_networks(base_folder, side_lengths, simulations, node_range, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for side_length in side_lengths:
        folder_name = f"graphs_test_{side_length}"
        if not os.path.exists(folder_name):
            print(f"Folder {folder_name} does not exist. Skipping...")
            continue
        
        for sim in range(1, simulations + 1):
            selected_files = {}
            
            for filename in os.listdir(folder_name):
                if filename.endswith(".graphml") and f"_sim{sim}" in filename:
                    parts = filename.split("-")
                    node_part = parts[1].split("_")[0]
                    node_count = int(node_part.replace("neighbors", ""))
                    
                    if node_count in node_range:
                        if node_count not in selected_files:
                            selected_files[node_count] = []
                        selected_files[node_count].append(filename)
            
            for node, files in selected_files.items():
                chosen_file = random.choice(files)
                src_path = os.path.join(folder_name, chosen_file)
                new_filename = f"{os.path.splitext(chosen_file)[0]}_{side_length}km.graphml"
                dest_path = os.path.join(output_folder, new_filename)
                shutil.copy(src_path, dest_path)
                print(f"Copied {chosen_file} -> {new_filename}")


def select_random_networks_from_extra(base_folder, node_range, num_networks_per_node, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    folder_name = "graphs_extra"  # Folder containing the extra networks
    if not os.path.exists(folder_name):
        print(f"Folder {folder_name} does not exist. Skipping...")
        return
    
    selected_files = {}

    # Go through all the files in the graphs_extra folder
    for filename in os.listdir(folder_name):
        if filename.endswith(".graphml"):
            parts = filename.split("-")
            node_part = parts[1].split("_")[0]
            node_count = int(node_part.replace("neighbors", ""))

            if node_count in node_range:
                if node_count not in selected_files:
                    selected_files[node_count] = []
                selected_files[node_count].append(filename)

    # For each node count, select 'num_networks_per_node' random networks (up to the available ones)
    for node, files in selected_files.items():
        files_to_select = random.sample(files, min(len(files), num_networks_per_node))  # Select up to 'num_networks_per_node' files

        for chosen_file in files_to_select:
            src_path = os.path.join(folder_name, chosen_file)
            new_filename = f"{os.path.splitext(chosen_file)[0]}_selected.graphml"
            dest_path = os.path.join(output_folder, new_filename)
            shutil.copy(src_path, dest_path)
            print(f"Copied {chosen_file} -> {new_filename}")

# # Define parameters
# side_lengths = [500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]  # Example side lengths
# simulations = 26  # Number of simulations
# node_range = range(81, 101)  # Nodes from 7 to 60
# output_folder = "test_dataset"

# # Run the selection
# select_random_networks(".", side_lengths, simulations, node_range, output_folder)

# Run the selection for graphs_extra folder
node_range = range(10, 101, 10)  # Nodes from 7 to 60
extra_output_folder = "graphs_extra_filtered"  # Output folder for the new selection
num_networks_per_node = 50  # Number of networks to select per node count
select_random_networks_from_extra(".", node_range, num_networks_per_node, extra_output_folder)