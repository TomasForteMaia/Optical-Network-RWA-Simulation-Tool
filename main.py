"""
main.py

This script contains the main logic for simulating and analyzing optical networks. 
It computes topological characteristics, path computations, power optimization, 
and link SNR for a variety of network configurations. The results are then processed 
and saved, including network capacity, blocking ratio, and other relevant metrics.

Author: Tomás Maia 96340 MEEC
Date: 21/02/2025
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import ticker to format ticks
import pandas as pd
import time # Test computation times
import os

from config import Config
from band_definition import compute_channel_frequencies, compute_relative_frequencies, define_bands
from modulation_ber_computation import ber_qam, ber_psk, snr_qam, snr_psk, compute_snr_thresholds, compute_gaussian_threshold
from optimization import precompute_links, perform_power_optimization_with_isrs, optimize_links
from noise_gain_computation import compute_isrs_gain_loss, compute_roadm_noise, compute_ase_noise_isrs, compute_post_amplifier_gain, compute_line_amplifier_gain
from path_computation import k_shortest_paths, compute_k_shortest_paths, path_distance
from traffic_handling import order_traffic_demands, prepare_available_wavelengths, assign_wavelength_with_snr_modulation_check, handle_traffic_demands_with_snr_modulation, handle_static_traffic, handle_progressive_traffic
from calculate_snr import compute_link_snr, calculate_snr_for_path
from network_analysis import visualize_random_network, calculate_topological_characteristics, display_routing_results, generate_random_demand

def compute_feature_statistics(features):
    """
    Compute statistics for each feature in the dataset.

    Parameters:
    - features: List of feature arrays.

    Returns:
    - DataFrame containing the statistics for each feature.
    """
    feature_names = [
        "num_nodes", "num_links", "min_link_length", "max_link_length", "avg_link_length",
        "var_link_length", "min_node_degree", "max_node_degree", "avg_node_degree",
        "var_node_degree", "network_diameter", "algebraic_connectivity"
    ]

    df = pd.DataFrame(features, columns=feature_names)

    statistics = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    statistics = statistics.rename(columns={
        "min": "Minimum",
        "25%": "1st Quartile",
        "50%": "Median",
        "mean": "Mean",
        "75%": "3rd Quartile",
        "max": "Maximum"
    })

    return statistics[["Minimum", "1st Quartile", "Median", "Mean", "3rd Quartile", "Maximum"]]

def plot_average_link_length(network_results, by_links=False):
    """
    Plot average link length as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    average_link_lengths = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        avg_link_length = result['avg_link_length']

        if count not in counts:
            counts[count] = []
            average_link_lengths[count] = []

        counts[count].append(count)
        average_link_lengths[count].append(avg_link_length)

    sorted_counts = sorted(counts.keys())
    avg_link_lengths = [np.mean(average_link_lengths[count]) for count in sorted_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_counts, avg_link_lengths, 'mo-', label='Avg Link Length')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Average Link Length (km)')
    plt.title('Average Link Length vs Number of Links' if by_links else 'Average Link Length vs Number of Nodes')
    plt.grid(True)
    plt.show()

def plot_network_capacity_vs_avg_link_length(network_results):
    """
    Plot a scatter plot for network capacity as a function of the average link length.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    avg_link_lengths = []
    network_capacities = []

    for result in network_results:
        avg_link_length = result['avg_link_length']
        network_capacity = result['network_capacity']
        avg_link_lengths.append(avg_link_length)
        network_capacities.append(network_capacity)

    plt.figure(figsize=(10, 6))
    plt.scatter(avg_link_lengths, network_capacities, c='blue', alpha=0.5)
    plt.xlabel('Average Link Length [km]')
    plt.ylabel('Network Capacity [Tbps]')
    plt.title('Network Capacity vs Average Link Length')
    plt.grid(True)
    plt.show()

def plot_avg_channel_capacity_vs_avg_link_length(network_results):
    """
    Plot a scatter plot for average channel capacity as a function of the average link length.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    avg_link_lengths = []
    avg_channel_capacities = []

    for result in network_results:
        avg_link_length = result['avg_link_length']
        avg_channel_capacity = result['average_channel_capacity']
        avg_link_lengths.append(avg_link_length)
        avg_channel_capacities.append(avg_channel_capacity)

    plt.figure()
    plt.scatter(avg_link_lengths, avg_channel_capacities, alpha=0.5)
    plt.xlabel('Average Link Length [km]')
    plt.ylabel('Average Channel Capacity [Gbps]')
    plt.title('Average Channel Capacity vs Average Link Length')
    plt.grid(True)
    plt.show()

def plot_blocking_probabilities_vs_avg_link_length(network_results):
    """
    Plot a scatter plot for blocking probabilities as a function of the average link length.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    avg_link_lengths = []
    blocking_probabilities = []

    for result in network_results:
        avg_link_length = result['avg_link_length']
        blocking_probability = result['blocking_ratio']
        avg_link_lengths.append(avg_link_length)
        blocking_probabilities.append(blocking_probability)

    plt.figure()
    plt.scatter(avg_link_lengths, blocking_probabilities, alpha=0.5)
    plt.xlabel('Average Link Length [km]')
    plt.ylabel('Blocking Probability')
    plt.title('Blocking Probability vs Average Link Length')
    plt.grid(True)
    plt.show()

def plot_average_snr_vs_avg_link_length(network_results):
    """
    Plot a scatter plot for average SNR as a function of the average link length.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    avg_link_lengths = []
    average_snrs = []

    for result in network_results:
        avg_link_length = result['avg_link_length']
        average_snr = result['average_snr']
        avg_link_lengths.append(avg_link_length)
        average_snrs.append(average_snr)

    plt.figure()
    plt.scatter(avg_link_lengths, average_snrs, c='purple', alpha=0.5)
    plt.xlabel('Average Link Length [km]')
    plt.ylabel('Average SNR [dB]')
    plt.title('Average SNR vs Average Link Length')
    plt.grid(True)
    plt.show()

def plot_blocked_due_to_snr_vs_avg_link_length(network_results):
    """
    Plot a scatter plot for blocked demands due to SNR as a function of the average link length.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    avg_link_lengths = []
    blocked_due_to_snr = []

    for result in network_results:
        avg_link_length = result['avg_link_length']
        blocked_snr = result['blocked_due_to_snr']
        avg_link_lengths.append(avg_link_length)
        blocked_due_to_snr.append(blocked_snr)

    plt.figure()
    plt.scatter(avg_link_lengths, blocked_due_to_snr, c='cyan', alpha=0.5)
    plt.xlabel('Average Link Length [km]')
    plt.ylabel('Blocked Demands Due to SNR')
    plt.title('Blocked Demands Due to SNR vs Average Link Length')
    plt.grid(True)
    plt.show()

def plot_blocked_due_to_wavelength_vs_avg_link_length(network_results):
    """
    Plot a scatter plot for blocked demands due to wavelength availability as a function of the average link length.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    avg_link_lengths = []
    blocked_due_to_wavelength = []

    for result in network_results:
        avg_link_length = result['avg_link_length']
        blocked_wavelength = result['blocked_due_to_wavelength']
        avg_link_lengths.append(avg_link_length)
        blocked_due_to_wavelength.append(blocked_wavelength)

    plt.figure()
    plt.scatter(avg_link_lengths, blocked_due_to_wavelength, c='orange', alpha=0.5)
    plt.xlabel('Average Link Length [km]')
    plt.ylabel('Blocked Demands Due to Wavelength')
    plt.title('Blocked Demands Due to Wavelength vs Average Link Length')
    plt.grid(True)
    plt.show()

def plot_network_capacity(network_results, by_links=False):
    """
    Plot network capacity as a function of the number of nodes or links using a box plot.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    capacities = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        capacity = result['network_capacity']

        if count not in counts:
            counts[count] = []
            capacities[count] = []

        counts[count].append(count)
        capacities[count].append(capacity)

    sorted_counts = sorted(counts.keys())
    sorted_capacities = [capacities[count] for count in sorted_counts]

    # Print statistics for each group (sorted_counts)
    for i, count in enumerate(sorted_counts):
        data = sorted_capacities[i]
        
        # Compute the statistics
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        median = np.median(data)
        mean = np.mean(data)
        min_value = np.min(data)
        max_value = np.max(data)
        std_dev = np.std(data)

        # Print out the statistics
        print(f"Statistics for {count} {'Links' if by_links else 'Nodes'}:")
        print(f"  Minimum: {min_value:.2f}")
        print(f"  1st Quartile (Q1): {Q1:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Mean: {mean:.2f}")
        print(f"  3rd Quartile (Q3): {Q3:.2f}")
        print(f"  Maximum: {max_value:.2f}")
        print(f"  Standard Deviation: {std_dev:.2f}")
        print("-" * 50)

    plt.figure()
    plt.boxplot(sorted_capacities, labels=sorted_counts, patch_artist=True)
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Network Capacity (Tbps)')
    plt.title('Network Capacity vs Number of Links' if by_links else 'Network Capacity vs Number of Nodes')
    plt.grid(True)
    plt.show()

    # Plot average network capacity
    avg_network_capacities = [np.mean(capacities[count]) for count in sorted_counts]
    plt.figure()
    plt.plot(sorted_counts, avg_network_capacities, 'go-', label='Avg Network Capacity')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Average Network Capacity (Tbps)')
    plt.title('Average Network Capacity vs Number of Links' if by_links else 'Average Network Capacity vs Number of Nodes')
    plt.grid(True)
    plt.show()

def plot_avg_channel_capacity(network_results, by_links=False):
    """
    Plot average channel capacity as a function of the number of nodes or links using a box plot.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    avg_channel_capacities = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        avg_channel_capacity = result['average_channel_capacity']

        if count not in counts:
            counts[count] = []
            avg_channel_capacities[count] = []

        counts[count].append(count)
        avg_channel_capacities[count].append(avg_channel_capacity)

    sorted_counts = sorted(counts.keys())
    sorted_avg_channel_capacities = [avg_channel_capacities[count] for count in sorted_counts]

    # Print statistics for each group (sorted_counts)
    for i, count in enumerate(sorted_counts):
        channel_capacity_data = sorted_avg_channel_capacities[i]

        # Compute statistics
        Q1 = np.percentile(channel_capacity_data, 25)
        Q3 = np.percentile(channel_capacity_data, 75)
        median = np.median(channel_capacity_data)
        mean = np.mean(channel_capacity_data)
        min_value = np.min(channel_capacity_data)
        max_value = np.max(channel_capacity_data)
        std_dev = np.std(channel_capacity_data)

        # Print out the statistics
        print(f"\nChannel Capacity Statistics for {count} {'Links' if by_links else 'Nodes'}:")
        print(f"  Minimum: {min_value:.2f} Gbps")
        print(f"  1st Quartile (Q1): {Q1:.2f} Gbps")
        print(f"  Median: {median:.2f} Gbps")
        print(f"  Mean: {mean:.2f} Gbps")
        print(f"  3rd Quartile (Q3): {Q3:.2f} Gbps")
        print(f"  Maximum: {max_value:.2f} Gbps")
        print(f"  Standard Deviation: {std_dev:.2f} Gbps")
        print("-" * 50)


    plt.figure()
    plt.boxplot(sorted_avg_channel_capacities, labels=sorted_counts, patch_artist=True)
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Average Channel Capacity (Gbps)')
    plt.title('Average Channel Capacity vs Number of Links' if by_links else 'Average Channel Capacity vs Number of Nodes')
    plt.grid(True)
    plt.show()

    # Plot average channel capacity
    avg_channel_capacities = [np.mean(avg_channel_capacities[count]) for count in sorted_counts]
    plt.figure()
    plt.plot(sorted_counts, avg_channel_capacities, 'go-', label='Avg Channel Capacity')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Average Channel Capacity (Gbps)')
    plt.title('Average Channel Capacity vs Number of Links' if by_links else 'Average Channel Capacity vs Number of Nodes')
    plt.grid(True)
    plt.show()

def plot_blocking_probabilities(network_results, by_links=False):
    """
    Plot average blocking probability and number of blocked demands as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    blocking_probabilities = {}
    blocked_demands = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        blocking_probability = result['blocking_ratio']
        blocked_demand = result['blocked_demands']

        if count not in counts:
            counts[count] = []
            blocking_probabilities[count] = []
            blocked_demands[count] = []

        counts[count].append(count)
        blocking_probabilities[count].append(blocking_probability)
        blocked_demands[count].append(blocked_demand)

    sorted_counts = sorted(counts.keys())
    sorted_blocking_probabilities = [blocking_probabilities[count] for count in sorted_counts]
    sorted_blocked_demands = [blocked_demands[count] for count in sorted_counts]

    # Print statistics for each group (sorted_counts)
    for i, count in enumerate(sorted_counts):
        blocking_data = sorted_blocking_probabilities[i]
        blocked_demand_data = sorted_blocked_demands[i]

        # Compute statistics for blocking probability
        Q1_blocking = np.percentile(blocking_data, 25)
        Q3_blocking = np.percentile(blocking_data, 75)
        median_blocking = np.median(blocking_data)
        mean_blocking = np.mean(blocking_data)
        min_blocking = np.min(blocking_data)
        max_blocking = np.max(blocking_data)
        std_dev_blocking = np.std(blocking_data)

        # Compute statistics for blocked demands
        Q1_blocked = np.percentile(blocked_demand_data, 25)
        Q3_blocked = np.percentile(blocked_demand_data, 75)
        median_blocked = np.median(blocked_demand_data)
        mean_blocked = np.mean(blocked_demand_data)
        min_blocked = np.min(blocked_demand_data)
        max_blocked = np.max(blocked_demand_data)
        std_dev_blocked = np.std(blocked_demand_data)

        # Print out the statistics
        print(f"\nBlocking Statistics for {count} {'Links' if by_links else 'Nodes'}:")
        print(f"  Blocking Probability:")
        print(f"    Minimum: {min_blocking:.4f}")
        print(f"    1st Quartile (Q1): {Q1_blocking:.4f}")
        print(f"    Median: {median_blocking:.4f}")
        print(f"    Mean: {mean_blocking:.4f}")
        print(f"    3rd Quartile (Q3): {Q3_blocking:.4f}")
        print(f"    Maximum: {max_blocking:.4f}")
        print(f"    Standard Deviation: {std_dev_blocking:.4f}")

        print(f"  Blocked Demands:")
        print(f"    Minimum: {min_blocked:.2f}")
        print(f"    1st Quartile (Q1): {Q1_blocked:.2f}")
        print(f"    Median: {median_blocked:.2f}")
        print(f"    Mean: {mean_blocked:.2f}")
        print(f"    3rd Quartile (Q3): {Q3_blocked:.2f}")
        print(f"    Maximum: {max_blocked:.2f}")
        print(f"    Standard Deviation: {std_dev_blocked:.2f}")
        print("-" * 50)

    fig, ax1 = plt.subplots()

    # Plot blocking probability (red line) first
    line1, = ax1.plot(sorted_counts, [np.mean(blocking_probabilities[count]) for count in sorted_counts], 'ro-', label='Avg. Blocking Probability')  
    ax1.set_xlabel('Number of Links' if by_links else 'Number of Nodes')
    ax1.set_ylabel('Average Blocking Probability')
    ax1.tick_params(axis='y')

    # Plot blocked demands (blue line) second
    ax2 = ax1.twinx()
    line2, = ax2.plot(sorted_counts, [np.mean(blocked_demands[count]) for count in sorted_counts], 'bo-', label='Avg. Blocked Demands')
    ax2.set_ylabel('Average Blocked Demands')
    ax2.tick_params(axis='y')

    # Add legends for both lines
    ax1.legend(handles=[line1], loc='upper left', bbox_to_anchor=(0.05, 0.95))
    ax2.legend(handles=[line2], loc='upper right', bbox_to_anchor=(0.95, 0.95))

    fig.tight_layout()
    plt.title('Blocking Probability and Blocked Demands vs Number of Links' if by_links else 'Blocking Probability and Blocked Demands vs Number of Nodes')
    plt.show()


def plot_average_snr(network_results, by_links=False):
    """
    Plot average SNR as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    average_snrs = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        average_snr = result['average_snr']

        if count not in counts:
            counts[count] = []
            average_snrs[count] = []

        counts[count].append(count)
        average_snrs[count].append(average_snr)

    sorted_counts = sorted(counts.keys())
    avg_snrs = [np.mean(average_snrs[count]) for count in sorted_counts]

    plt.plot(sorted_counts, avg_snrs, 'go-', label='Avg SNR')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Average SNR (dB)')
    plt.title('Average SNR vs Number of Links' if by_links else 'Average SNR vs Number of Nodes')
    plt.grid(True)
    plt.show()


def plot_blocked_due_to_snr(network_results, by_links=False):
    """
    Plot number of blocked demands due to SNR as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    blocked_due_to_snr = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        blocked_snr = result['blocked_due_to_snr']

        if count not in counts:
            counts[count] = []
            blocked_due_to_snr[count] = []

        counts[count].append(count)
        blocked_due_to_snr[count].append(blocked_snr)

    sorted_counts = sorted(counts.keys())
    avg_blocked_snr = [np.mean(blocked_due_to_snr[count]) for count in sorted_counts]

    plt.plot(sorted_counts, avg_blocked_snr, 'co-', label='Blocked Due to SNR')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Blocked Demands Due to SNR')
    plt.title('Blocked Demands Due to SNR vs Number of Links' if by_links else 'Blocked Demands Due to SNR vs Number of Nodes')
    plt.grid(True)
    plt.show()


def plot_blocked_due_to_wavelength(network_results, by_links=False):
    """
    Plot number of blocked demands due to wavelength availability as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    blocked_due_to_wavelength = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        blocked_wavelength = result['blocked_due_to_wavelength']

        if count not in counts:
            counts[count] = []
            blocked_due_to_wavelength[count] = []

        counts[count].append(count)
        blocked_due_to_wavelength[count].append(blocked_wavelength)

    sorted_counts = sorted(counts.keys())
    avg_blocked_wavelength = [np.mean(blocked_due_to_wavelength[count]) for count in sorted_counts]

    plt.plot(sorted_counts, avg_blocked_wavelength, 'yo-', label='Blocked Due to Wavelength')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Blocked Demands Due to Wavelength')
    plt.title('Blocked Demands Due to Wavelength vs Number of Links' if by_links else 'Blocked Demands Due to Wavelength vs Number of Nodes')
    plt.grid(True)
    plt.show()

def plot_successful_connections(network_results, by_links=False):
    """
    Plot the number of successful connections as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    successful_connections = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        total_demands = result['traffic_demands']
        blocked_demands = result['blocked_demands']
        successful = total_demands - blocked_demands

        if count not in counts:
            counts[count] = []
            successful_connections[count] = []

        counts[count].append(count)
        successful_connections[count].append(successful)

    sorted_counts = sorted(counts.keys())
    avg_successful_connections = [np.mean(successful_connections[count]) for count in sorted_counts]

    plt.plot(sorted_counts, avg_successful_connections, 'go-', label='Successful Connections')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Successful Connections')
    plt.title('Successful Connections vs Number of Links' if by_links else 'Successful Connections vs Number of Nodes')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_longest_shortest_path_length(network_results, by_links=False):
    """
    Plot the length of the longest shortest-path as a function of the number of nodes or links.

    Parameters:
    - network_results: List of dictionaries containing network results.
    - by_links: Boolean flag indicating whether to plot by the number of links.
    """
    counts = {}
    longest_shortest_paths = {}

    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        longest_shortest_path = result['longest_shortest_path']

        if count not in counts:
            counts[count] = []
            longest_shortest_paths[count] = []

        counts[count].append(count)
        longest_shortest_paths[count].append(longest_shortest_path)

    sorted_counts = sorted(counts.keys())
    avg_longest_shortest_paths = [np.mean(longest_shortest_paths[count]) for count in sorted_counts]

    plt.figure()
    plt.plot(sorted_counts, avg_longest_shortest_paths, 'co-', label='Longest Shortest-Path Length')
    plt.xlabel('Number of Links' if by_links else 'Number of Nodes')
    plt.ylabel('Longest Shortest-Path Length (km)')
    plt.title('Longest Shortest-Path Length vs Number of Links' if by_links else 'Longest Shortest-Path Length vs Number of Nodes')
    plt.grid(True)
    plt.show()

def plot_link_congestion_by_nodes_or_links(network_results, by_links=False):
    """
    Plot link congestion for one network per node or link count.

    Parameters:
    - network_results: List of network results, each containing link congestion data.
    - by_links: If True, group by number of links; if False, group by number of nodes.
    """

    # Initialize a dictionary to group networks by node/link count
    grouped_results = {}

    # Group the results by number of nodes or links
    for result in network_results:
        count = result['num_links'] if by_links else result['num_nodes']
        
        # Initialize list for a new group
        if count not in grouped_results:
            grouped_results[count] = []

        # Add the network result to the group
        grouped_results[count].append(result)

    # Plot the link congestion for each group
    for count, congestion_list in grouped_results.items():
        # Only pick one network from each group (the first one for simplicity)
        selected_network = congestion_list[0]  # Or any other logic to select a network

        # Extract link congestion data from the selected network
        link_congestion = selected_network.get('link_saturation', {})

        if len(link_congestion) == 0:
            continue  # Skip if no congestion data

        # Filter links to include only (u, v) where u < v
        filtered_links = [(u, v) for (u, v) in link_congestion.keys() if u < v]

        if not filtered_links:
            continue  # Skip if no valid links remain

        # Prepare data for plotting
        congestion_values = [link_congestion[link][0] * 100 for link in filtered_links]  # Convert to percentage
        all_links_str = [f"{u}-{v}" for u, v in filtered_links]

        # Compute and print the average link congestion
        avg_congestion = sum(congestion_values) / len(congestion_values)
        print(f"Average link congestion for {count} {'nodes' if not by_links else 'links'}: {avg_congestion:.2f}%")

        # Plot the congestion for the selected network
        plt.figure(figsize=(10, 6))
        plt.bar(all_links_str, congestion_values, alpha=0.7)

        # Customizations
        plt.title(f"Link Congestion for {count} {'nodes' if not by_links else 'links'}", fontsize=16)
        plt.xlabel("Links")
        plt.ylabel("Congestion [%]")
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_average_link_length_scatter(network_results):
    """
    Plot a scatter plot with the average link length for each graph.

    Parameters:
    - network_results: List of dictionaries containing network results.
    """
    folder_lengths = []
    avg_link_lengths = []

    for result in network_results:
        side_length = result['side_length']
        avg_link_length = result['avg_link_length']
        folder_lengths.append(side_length)
        avg_link_lengths.append(avg_link_length)

    plt.figure()
    plt.scatter(folder_lengths, avg_link_lengths, c='blue', alpha=0.5)
    plt.xlabel('Side Length of 2D Plane (km)')
    plt.ylabel('Average Link Length (km)')
    plt.title('Average Link Length vs Side Length of 2D Plane')
    plt.grid(True)
    plt.show()

def plot_blocking_vs_capacity(mc_results, bands_to_analyze, graph_name):
    """
    Plot blocking probability vs. capacity for progressive traffic.

    Parameters:
    - mc_results: Dictionary of Monte Carlo results with blocking_probs and capacities.
    - bands_to_analyze: List of bands (e.g., ["C", "Super C", "Super C + Super L"]).
    - graph_name: Name of the graph/network being analyzed.
    """
    plt.figure()  # Reduced figure size for a tighter layout
    
    for band in bands_to_analyze:
        # Pad sequences to the same length
        max_lengths = {
            "blocking": max(len(blocking_probs) for blocking_probs in mc_results[band]["blocking_probs"]),
            "capacity": max(len(capacities) for capacities in mc_results[band]["capacities"]),
        }

        # padded_blocking_probs = np.array([
        #     np.pad(blocking_probs, (0, max_lengths["blocking"] - len(blocking_probs)), constant_values=np.nan)
        #     for blocking_probs in mc_results[band]["blocking_probs"]
        # ])
        padded_blocking_probs = np.array([
            np.pad(np.array(blocking_probs, dtype=float),
                   (0, max_lengths["blocking"] - len(blocking_probs)), 
                   constant_values=np.nan)
            for blocking_probs in mc_results[band]["blocking_probs"]
        ])

        # padded_capacities = np.array([
        #     np.pad(capacities, (0, max_lengths["capacity"] - len(capacities)), constant_values=np.nan)
        #     for capacities in mc_results[band]["capacities"]
        # ])
        padded_capacities = np.array([
            np.pad(np.array(capacities, dtype=float), 
                   (0, max_lengths["capacity"] - len(capacities)), 
                   constant_values=np.nan)
            for capacities in mc_results[band]["capacities"]
        ])
        
        avg_blocking_probs = np.nanmean(padded_blocking_probs, axis=0)
        avg_capacities_tbps = np.nanmean(padded_capacities, axis=0) / 1e3  # Convert to Tbps
        
        plt.plot(avg_capacities_tbps, avg_blocking_probs, label=f"{band} Band", linewidth=2)  # Thicker lines

    clean_name = graph_name.replace(".graphml", "")

    plt.xscale("linear")
    plt.yscale("log")
    plt.xlim(100, 800)   # Set x-axis limits
    plt.ylim(1e-4, 1e-1)  # Blocking probabilities range
    #plt.ylim(1e-4, 8e-1)
    plt.yticks([1e-4, 1e-3, 1e-2, 1e-1], labels=[r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$"])
    plt.xlabel("Network Capacity [Tbps]")
    plt.ylabel("Blocking Probability")
    plt.title(f"Blocking Probability vs Capacity ({clean_name})", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="major", linestyle="-", linewidth=0.7)  # Continuous major grid lines
    plt.tight_layout()
    plt.show()

def plot_link_congestion(mc_results, bands_to_analyze, graph_name):
    """
    Plot link congestion (average link saturation) for progressive traffic.

    Parameters:
    - mc_results: Dictionary of Monte Carlo results with link_saturation data.
    - bands_to_analyze: List of bands (e.g., ["C", "Super C", "Super C + Super L"]).
    - graph_name: Name of the graph/network being analyzed.
    """
    # Filter links to include only (u, v) where u < v
    all_links = list(next(iter(mc_results.values()))["link_saturation"].keys())
    filtered_links = [(u, v) for u, v in all_links if u < v]

    # Prepare data for plotting
    num_bands = len(bands_to_analyze)
    num_links = len(filtered_links)
    bar_width = 0.8 / num_bands  # Adjust bar width to fit multiple bands
    x_positions = np.arange(num_links)  # X positions for the links

    plt.figure()  # Set figure size for better readability

    for i, band in enumerate(bands_to_analyze):
        # Extract saturation for the filtered links
        link_saturation = mc_results[band]["link_saturation"]
        saturation_percentages = [
            link_saturation[link] * 100 for link in filtered_links
        ]  # Convert to percentage

        # Plot bars for the current band with offset to group bars
        plt.bar(
            x_positions + i * bar_width,
            saturation_percentages,
            width=bar_width,
            label=f"{band} Band",
            alpha=0.7,
        )

    clean_name = graph_name.replace(".graphml", "")

    # Customize the plot
    plt.title(f"Average Link Congestion ({clean_name})", fontsize=16)
    plt.xlabel("Links")
    plt.ylabel("Congestion [%]")
    plt.xticks(
        x_positions + (num_bands - 1) * bar_width / 2,  # Center labels
        [f"{u}-{v}" for u, v in filtered_links],
        rotation=90,
        fontsize=10,
    )
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)  # Add grid for y-axis
    plt.tight_layout()
    plt.show()

def plot_static_snap_analysis(static_aggregated_results, graph_name):
    """
    Function to analyze and plot results for the static traffic scenario.

    Parameters:
        static_aggregated_results (list): Collected results for static traffic (per network).
        graph_name (str): Name of the network graph.
    """
    print(f"\n--- PLOTTING STATIC TRAFFIC RESULTS FOR {graph_name} ---")

    # Extract metrics from aggregated results
    avg_bit_rates = [res["avg_bit_rate"] for res in static_aggregated_results]
    blocking_ratios = [res["blocking_ratio"] for res in static_aggregated_results]
    num_blocked = [res["num_blocked"] for res in static_aggregated_results]

    # Calculate statistics
    mean_bit_rate = np.mean(avg_bit_rates)
    std_bit_rate = np.std(avg_bit_rates)
    mean_blocking_ratio = np.mean(blocking_ratios)
    mean_num_blocked = np.mean(num_blocked)

    # Print average statistics
    print(f"\n--- NETWORK-WIDE RESULTS (Static Traffic) ---")
    print(f"Average Bit Rate per LP: {mean_bit_rate:.2f} Tbps (±{std_bit_rate:.2f})")
    print(f"Average Blocking Ratio: {mean_blocking_ratio:.4f}")
    print(f"Average Number of Blocked Demands: {mean_num_blocked:.2f}")

    # Plot: Distribution of Average Bit Rate per LP
    plt.figure(figsize=(10, 6))
    count, bins, _ = plt.hist(avg_bit_rates, bins=20, density=True, alpha=0.6, color='b', label="Simulated PDF")
    mean_label = f"Mean: {mean_bit_rate:.2f} Tbps"
    plt.axvline(mean_bit_rate, color='r', linestyle='--', label=mean_label)
    plt.title(f"PDF of Average Bit Rate per LP ({graph_name})")
    plt.xlabel("Average Bit Rate [Gbps]")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot: Convergence of Mean and Standard Deviation
    iteration_range = np.arange(1, len(avg_bit_rates) + 1)
    mean_values = [np.mean(avg_bit_rates[:n]) for n in iteration_range]
    std_values = [np.std(avg_bit_rates[:n]) for n in iteration_range]

    plt.figure(figsize=(10, 6))
    plt.plot(iteration_range, mean_values, label="Mean Bit Rate (Tbps)", color="blue")
    plt.plot(iteration_range, std_values, label="Std. Dev. Bit Rate (Tbps)", color="green")
    plt.xlabel("Monte Carlo Iterations")
    plt.ylabel("Value [Tbps]")
    plt.title(f"Convergence of Mean and Std. Dev. ({graph_name})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def append_to_npy(file_name, new_data):
    if os.path.exists(file_name):
        existing_data = np.load(file_name, allow_pickle=True)
        combined_data = np.concatenate((existing_data, new_data), axis=0)
    else:
        combined_data = new_data
    np.save(file_name, combined_data)


# --------------------------- MAIN LOGIC ---------------------------------- #

# Set pandas display options to avoid truncation
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

if __name__ == "__main__":

    contiguous = False # Boolean flag indicating if the bands should be contiguous.

    config = Config()
    config.initialize_bands(contiguous=contiguous) 
    config.initialize_thresholds()
    config.load_networks(["real_networks"])

    # Print the simulation parameters
    config.print_parameters()

    features = []  # Features array
    labels = []    # Labels array
    network_results = []          

    # Your folders list
    folders = ["graphs_30_750_750", "graphs_30_1000_1000", "graphs_30_1500_1500", "graphs_30_2000_2000", 
               "graphs_30_2500_2500", "graphs_30_3000_3000", "graphs_30_3500_3500", "graphs_30_4000_4000", 
               "graphs_30_4500_4500", "graphs_30_5000_5000", "graphs_30_6000_6000", "graphs_30_7000_7000", "graphs_30_8000_8000", "graphs_30_9000_9000"]


    # config.networks = config.filter_networks_by_nodes(
    #     folder="graphs_links",
    #     node_counts=list(range(10, 101, 10)),  # Nodes from 10 to 100 in steps of 10
    #     num_networks_per_count=50  # Adjust this number as needed
    # )        

    # link_counts = range(80, 141, 5)

    # Load networks from the specified folders
    # config.networks = config.load_and_filter_networks_by_links(folders=["graphs_links"], num_networks_per_link=50, link_counts=link_counts)  # Adjust number as needed

    total_computation_start_time = time.perf_counter() 
    network_count = len(config.networks)  # Total networks under analysis
    # network_count = sum(len(graphs) for graphs in config.networks.values())  # Total networks under analysis
    idx = 0

    # Iterate through the folders and their graphs
    # for folder, graphs in config.networks.items():
    #     for graph_name, G in graphs:
    for graph_name, G in config.networks.items():
        idx += 1

        current_time = time.perf_counter() - total_computation_start_time  # Get elapsed time

        print(f'\nNetworks Analyzed: {idx}/{network_count}')
        print(f"Elapsed Simulation Time: {current_time:.2f} seconds")  # Print elapsed time
        print(f"\nAnalyzing Graph: {graph_name} ({len(G.nodes)} Nodes)")
        topology_characteristics = calculate_topological_characteristics(G, config.verbose)
        # visualize_random_network(G, title=f"{graph_name} Network Topology")

        # Start the total computation timer
        total_start_time = time.perf_counter()

        # -------------------------------------------------------------------------------- #
        #  STEP 1: Paths Computation, Channel Power Optimization And Link SNR Computation  #
        # -------------------------------------------------------------------------------- #

        print("\nSTEP 1: Compute Shortest Paths for All Node-Pairs, Perform Power Optimization for all Links and Compute Link SNR")

        # Compute k-shortest paths for traffic demands (full-mesh topology)
        path_start_time = time.perf_counter()
        traffic_matrix = None  # Full-mesh topology
        all_pairs_shortest_paths = compute_k_shortest_paths(G, config.k_paths, traffic_matrix)
        path_end_time = time.perf_counter()
        path_computation_time = path_end_time - path_start_time

        # Start timing the link optimization
        optimization_start_time = time.perf_counter()
        precomputed_link_data = precompute_links(
            G, config
        )

        optimization_end_time = time.perf_counter()
        optimization_time = optimization_end_time - optimization_start_time

        if config.progressive_traffic:
            mc_results = handle_progressive_traffic(
                G, all_pairs_shortest_paths, precomputed_link_data, config
            )
            plot_blocking_vs_capacity(mc_results, config.bands, graph_name)

            # Plot Link Congestion (Progressive Traffic)
            plot_link_congestion(mc_results, config.bands, graph_name)

        else:
            static_results = handle_static_traffic(
                G, all_pairs_shortest_paths, precomputed_link_data, config
            )

            total_end_time = time.perf_counter()  # End the total computation timer
            total_time = total_end_time - total_start_time

            # Add results for NMC=1
            if config.NMC == 1:

                if config.verbose:
                    static_results[0]["path_computation_time"] = path_computation_time
                    static_results[0]["optimization_time"] = optimization_time
                    static_results[0]["computation_time"] = total_time
                    static_results[0]["num_nodes"] = topology_characteristics["num_nodes"]  # Add number of nodes to results
                    static_results[0]["num_links"] = topology_characteristics["num_links"]  # Add number of links to results
                    static_results[0]["avg_link_length"] = topology_characteristics["avg_link_length"] # Add average link length to results
                    network_results.append(static_results[0]) 

                    # Print detailed results
                    print(f"\n--- RESULTS FOR {graph_name} ---")
                    print(f"  Band Used: {static_results[0]['band_used']}")
                    print(f"  Total Computation Time: {static_results[0]['computation_time']:.2f} seconds")
                    print(f"  Path Computation Time: {static_results[0]['path_computation_time']:.2f} seconds")
                    print(f"  Link Optimization Time: {static_results[0]['optimization_time']:.2f} seconds")
                    print(f"  Sorting Time: {static_results[0]['sorting_time']:.2f} seconds")
                    print(f"  Assignment Time: {static_results[0]['assignment_time']:.2f} seconds")
                    print(f"  Average SNR: {static_results[0]['average_snr']:.2f} dB")
                    print(f"  Average Channel Capacity: {static_results[0]['average_channel_capacity']:.2f} Gbps")
                    print(f"  Network Capacity: {static_results[0]['network_capacity']:.2f} Tbps")
                    print(f"  Blocking Ratio: {static_results[0]['blocking_ratio']:.4f}")
                    print(f"  Number of Traffic Demands: {static_results[0]['traffic_demands']}")
                    print(f"  Number of Blocked Demands: {static_results[0]['blocked_demands']}")
                    print(f"  Blocked due to SNR: {static_results[0]['blocked_due_to_snr']}")
                    print(f"  Blocked due to Wavelength Availability: {static_results[0]['blocked_due_to_wavelength']}")
                    print(f"  Number of Required Wavelengths: {static_results[0]['required_wavelengths']}")
                    print(f"  Used Wavelengths Indices: {static_results[0]['used_wavelengths']}")

                    modulation_stats = static_results[0]['modulation_stats']

                    print("\n--- Modulation Usage Statistics ---")
                    for mod, stats in modulation_stats.items():
                        print(f"Modulation: {mod}")
                        print(f"  Count: {stats['count']}")
                        # if "details" in stats:
                        #     for detail in stats["details"]:
                        #         print(f"   SNR: {(10 * np.log10(detail['adjusted_snr'])):.2f} dB, Wavelength: {detail['wavelength']}")

                    # Plot modulation usage if non-Gaussian
                    if not config.use_gaussian_modulation:
                        mod_stats = static_results[0]["modulation_stats"]
                        modulation_formats = list(mod_stats.keys())
                        counts = [mod_stats[mod]["count"] for mod in modulation_formats]

                        plt.figure(figsize=(6, 3))  # Smaller width and height
                        bars = plt.bar(modulation_formats, counts, color="skyblue")

                        plt.ylabel("Allocated Demands")
                        plt.xlabel("Modulation Format")

                        # Increase y-axis limit slightly to fit labels
                        max_count = max(counts)
                        offset = 0.05 * max_count
                        plt.ylim(0, max_count + offset * 3)  # Add extra space at the top

                        # Add count labels slightly above each bar
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                                     f'{height}', ha='center', va='bottom')

                        plt.tight_layout()
                        plt.show()

                if not config.verbose:
                    topology_features = list(topology_characteristics.values())  # Extract features

                    # Add topology and simulation results
                    for result in static_results:
                        features.append(list(topology_characteristics.values()))  # Add features
                        labels.append([
                            result["network_capacity"],          # Network capacity (Tbps)
                            result["blocking_ratio"],            # Blocking ratio
                            result["average_channel_capacity"],  # Avg. channel capacity (Gbps)
                        ])

                        # Print computed labels when appending
                        print(f"Appended Labels for {graph_name}:")
                        print(f"  Network Capacity: {result['network_capacity']:.2f} Tbps")
                        print(f"  Average Channel Capacity: {result['average_channel_capacity']:.2f} Gbps")
                        print(f"  Blocking Ratio: {result['blocking_ratio']:.4f}\n")

    total_computation_end_time = time.perf_counter()
    total_computation_time = total_computation_end_time - total_computation_start_time
    print(f"\nTotal Computation Time: {total_computation_time:.2f} seconds")

    # Save only if verbose is False
    if not config.verbose:
        # Append new features and labels to existing files
        append_to_npy("features_gaussian_test.npy", np.array(features))
        append_to_npy("labels_gaussian_test.npy", np.array(labels))

        print("\nFeatures and labels appended to 'features.npy' and 'labels.npy'")

    plot_link_congestion_by_nodes_or_links(network_results, by_links=False) 

    # Plot network capacity vs. number of nodes/links
    plot_network_capacity(network_results, by_links=True)
    plot_avg_channel_capacity(network_results, by_links=True)
    plot_blocking_probabilities(network_results, by_links=True)
    plot_average_snr(network_results, by_links=True)
    plot_average_link_length(network_results, by_links=True)
    plot_blocked_due_to_snr(network_results, by_links=True)
    plot_blocked_due_to_wavelength(network_results, by_links=True)
    plot_successful_connections(network_results, by_links=True)
    plot_longest_shortest_path_length(network_results, by_links=True)
    

    # Plot scatter plots for network capacity and average channel capacity as a function of the average link length
    plot_network_capacity_vs_avg_link_length(network_results)
    plot_avg_channel_capacity_vs_avg_link_length(network_results)
    plot_blocking_probabilities_vs_avg_link_length(network_results)
    plot_average_snr_vs_avg_link_length(network_results)
    plot_blocked_due_to_snr_vs_avg_link_length(network_results)
    plot_blocked_due_to_wavelength_vs_avg_link_length(network_results)

# # Define BER values to analyze
# ber_values = [1e-3, 1e-2, 3.3e-2]

# # Dictionary to hold blocking ratios for each network over BERs
# blocking_ratios_by_network = {graph_name: [] for graph_name in config.networks}

# print("\nStarting Per-Network BER Sensitivity Analysis...\n")

# for ber in ber_values:
#     print(f"\n--- Running Static Traffic for BER = {ber:.0e} ---")

#     # Update BER and re-initialize thresholds
#     config.ber = ber
#     config.initialize_thresholds()

#     for graph_name, G in config.networks.items():
#         print(f"\nAnalyzing Graph: {graph_name} with BER = {ber:.0e}")

#         # Compute k-shortest paths
#         traffic_matrix = None
#         all_pairs_shortest_paths = compute_k_shortest_paths(G, config.k_paths, traffic_matrix)

#         # Precompute link data
#         precomputed_link_data = precompute_links(G, config)

#         # Run static traffic simulation
#         static_results = handle_static_traffic(G, all_pairs_shortest_paths, precomputed_link_data, config)

#         if static_results:
#             blocking_ratio = static_results[0]["blocking_ratio"]
#             blocking_ratios_by_network[graph_name].append(blocking_ratio)

#             # Print the results for the current network and BER
#             print(f"  Blocking Ratio: {blocking_ratio * 100:.2f}%")
#             print(f"  Blocked due to SNR: {static_results[0]['blocked_due_to_snr']}")
#             print(f"  Blocked due to Wavelength Availability: {static_results[0]['blocked_due_to_wavelength']}")
#         else:
#             blocking_ratios_by_network[graph_name].append(None)  # In case of failure

# # --- Plotting Blocking Ratio vs BER for all networks ---

# # plt.figure(figsize=(10, 6))
# for network, blocking_ratios in blocking_ratios_by_network.items():
#     # Strip ".graphml" from network name
#     clean_name = network.replace(".graphml", "")
#     plt.plot(
#         ber_values,
#         [br * 100 if br is not None else None for br in blocking_ratios],
#         marker="o",
#         label=clean_name
#     )

# plt.xlabel("BER")
# plt.ylabel("Blocking Ratio [%]")
# plt.title("Blocking Ratio vs BER for Different Networks")

# # Format BER values as 'a×10^b'
# ber_labels = [r"$1 \times 10^{-3}$", r"$1 \times 10^{-2}$", r"$3.3 \times 10^{-2}$"]

# plt.xticks(ber_values, ber_labels)

# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()