"""
Filename: traffic_handling.py
This module implements wavelength assignment and SNR verification for an optical network.

Author: Tomás Maia 96340 MEEC
Date: 10/02/2025

Description:
This file contains functions for managing and simulating traffic demands in optical networks. 
The functions handle both static and progressive traffic scenarios with the ability to assign wavelengths, 
account for modulation formats and SNR thresholds, and calculate network performance metrics like blocking ratio and capacity.
"""

import numpy as np
import random
import time

from path_computation import path_distance
from calculate_snr import calculate_snr_for_path
from network_analysis import generate_random_demand

def order_traffic_demands(G, all_pairs_shortest_paths, order_strategy="shortest-first"):
    """
    Order traffic demands based on the specified strategy, ensuring bidirectional demands are consecutive.

    Parameters:
    - G: NetworkX graph representing the network topology.
    - all_pairs_shortest_paths: A dictionary with keys as (source, target) and values as lists of paths.
    - order_strategy: Strategy to order the traffic demands. Options: "shortest-first", "longest-first", "random".

    Returns:
    - traffic_demands: A sorted list of traffic demands with bidirectional demands grouped consecutively.
    """
    traffic_demands = []

    # Step 1: Append each demand and its reverse (source, target) and (target, source with reversed paths)
    for (source, target), paths in all_pairs_shortest_paths.items():
        shortest_path_len = path_distance(G, paths[0])  # Length of shortest path
        traffic_demands.append((source, target, paths, shortest_path_len)) # Append the original direction
        reversed_paths = [list(reversed(path)) for path in paths] # Append the reverse direction, with paths reversed
        traffic_demands.append((target, source, reversed_paths, shortest_path_len))
        

    # Step 2: Sort based on the chosen strategy, keeping bidirectional demands together
    if order_strategy == "shortest-first":
        traffic_demands.sort(key=lambda x: x[3])
    elif order_strategy == "longest-first":
        traffic_demands.sort(key=lambda x: x[3], reverse=True)
    elif order_strategy == "random":
        # Group traffic demands into unique bidirectional pairs and shuffle each as a unit
        unique_pairs = list(set((min(d[0], d[1]), max(d[0], d[1])) for d in traffic_demands))
        random.shuffle(unique_pairs)  # Randomly shuffle the unique pairs
        shuffled_traffic_demands = []
        for src, dst in unique_pairs:
            # Add both directions of each demand consecutively
            for demand in traffic_demands:
                if {demand[0], demand[1]} == {src, dst}:
                    shuffled_traffic_demands.append(demand)
        traffic_demands = shuffled_traffic_demands
    else:
        raise ValueError("Unknown traffic sorting strategy.")

    return traffic_demands

def prepare_available_wavelengths(G, num_channels):
    """
    Prepare the available wavelengths dictionary for all links in the network.

    Parameters:
    - G: NetworkX graph representing the network topology.
    - num_channels: Number of wavelengths (channels) available per link.

    Returns:
    - available_wavelengths: Dictionary where keys are links (u, v) and values are lists of Boolean values 
      indicating whether each wavelength is available.
    """
    available_wavelengths = {}

    # Initialize wavelength availability for all bidirectional links
    for u, v in G.edges:
        available_wavelengths[(u, v)] = [True] * num_channels  # Forward direction
        available_wavelengths[(v, u)] = [True] * num_channels  # Reverse direction

    return available_wavelengths

def assign_wavelength_with_snr_modulation_check(
    path, available_wavelengths, config, params, assignment_strategy="first-fit",
    use_gaussian_modulation=True, modulation_formats=None, gaussian_threshold=None
):
    """
    Assign a wavelength to a path and verify SNR requirements immediately, with safety margin consideration.

    Parameters:
    - path: The path (list of nodes) for the demand.
    - available_wavelengthshandle_static_traffic: Dictionary tracking available wavelengths on each link.
    - config: Configuration object containing system parameters.
    - params: Dictionary of system parameters.
    - assignment_strategy: Strategy for assigning wavelengths. Options: "first-fit", "most-used", "least-used", "random".
    - use_gaussian_modulation: Boolean flag to toggle between Gaussian and real modulations.
    - modulation_formats: Dictionary of modulation formats and their SNR thresholds.
    - gaussian_threshold: Minimum SNR threshold for Gaussian modulation (only used if `use_gaussian_modulation=True`).

    Returns:
    - assigned_wavelength: The wavelength assigned to the path, or None if no wavelength is available or SNR is insufficient.
    - snr: The original SNR for the lightpath if assigned, or None if assignment failed.
    - adjusted_snr: The SNR after subtracting the safety margin, or None if assignment failed.
    - capacity: The capacity (Gbps) for the lightpath if assigned, or None if assignment failed.
    - modulation_used: Modulation format used (Gaussian or real).
    - block_reason: "no_wavelength_available" or "insufficient_snr".
    - safety_margin: The computed safety margin for the path.
    - n_roadms: Number of ROADMs traversed on the path.
    - n_olas: Number of spans (optical amplifiers) traversed on the path.
    """
    channels = params["num_channels"]
    SNR_links = params["SNR_links"]

    # Step 1: Determine wavelength availability for the entire path.
    path_wavelength_availability = [True] * channels

    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])
        path_wavelength_availability = [
            path_wavelength_availability[j] and available_wavelengths[link][j]
            for j in range(channels)
        ]

    # Step 2: Define wavelength selection order based on assignment strategy.
    if assignment_strategy == "first-fit":
        wavelength_order = range(channels)  # Default order by index (first-fit)
    elif assignment_strategy == "inverse-first-fit":
        wavelength_order = range(channels - 1, -1, -1)  # Reverse order (inverse first-fit)
    elif assignment_strategy == "most-used":
        wavelength_usage_counts = [sum(1 for link in available_wavelengths if not available_wavelengths[link][w]) for w in range(channels)]
        wavelength_order = sorted(range(channels), key=lambda w: wavelength_usage_counts[w], reverse=True)
    elif assignment_strategy == "least-used":
        wavelength_usage_counts = [sum(1 for link in available_wavelengths if not available_wavelengths[link][w]) for w in range(channels)]
        wavelength_order = sorted(range(channels), key=lambda w: wavelength_usage_counts[w])
    elif assignment_strategy == "random":
        wavelength_order = list(range(channels))
        random.shuffle(wavelength_order)
    else:
        raise ValueError("Unknown wavelength assignment strategy.")

    if not any(path_wavelength_availability):
        # No wavelengths are available
        return None, None, None, None, None, "no_wavelength_available", None, None, None

    # Step 3: Calculate the safety margin
    N_ROADMs = len(path)  # Include source (add) and destination (drop) nodes
    N_OLAs = sum(params['N_spans'][(path[i], path[i + 1])] for i in range(len(path) - 1))
    safety_margin = 2 + 0.05 * (N_OLAs + N_ROADMs)

    # Step 4: Precompute the SNR for the path
    if config.use_gaussian_modulation:
        total_SNR = calculate_snr_for_path(path, config, params, SNR_links["Gaussian"], modulation="Gaussian")
        precomputed_SNRs = {"Gaussian": total_SNR}
    else:
        precomputed_SNRs = {}
        for mod_format in modulation_formats:
            precomputed_SNRs[mod_format] = calculate_snr_for_path(path, config, params, SNR_links[mod_format], modulation=mod_format)

    # Step 5: Attempt to assign a wavelength.
    for w in wavelength_order:
        if path_wavelength_availability[w]:
            # Check SNR based on modulation type
            if config.use_gaussian_modulation:
                # Gaussian modulation case
                snr = precomputed_SNRs["Gaussian"][w]
                adjusted_snr_db = 10 * np.log10(snr) - safety_margin
                adjusted_snr_linear = 10 ** (adjusted_snr_db / 10)  # Convert adjusted SNR back to linear

                if adjusted_snr_db >= gaussian_threshold:  # Compare against threshold in dB
                    capacity = 2 * config.Rs * (1 / (1 + params['OH'])) * np.log2(1 + adjusted_snr_linear) / 1e9  # Shannon capacity in Gbps
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        available_wavelengths[link][w] = False
                    return w, snr, adjusted_snr_linear, capacity, "Gaussian", None, safety_margin, N_ROADMs, N_OLAs
            else:
                # Check each modulation format in descending order of capacity
                for mod_format, threshold in sorted(config.modulation_formats_with_thresholds.items(), key=lambda x: config.modulation_bit_rates[x[0]], reverse=True):
                    snr = precomputed_SNRs[mod_format][w]
                    adjusted_snr_db = 10 * np.log10(snr) - safety_margin
                    adjusted_snr_linear = 10 ** (adjusted_snr_db / 10)  # Convert adjusted SNR back to linear

                    if adjusted_snr_db >= threshold:  # Compare against the format's threshold in dB
                        capacity = config.modulation_bit_rates[mod_format]
                        for i in range(len(path) - 1):
                            link = (path[i], path[i + 1])
                            available_wavelengths[link][w] = False
                        return w, snr, adjusted_snr_linear, capacity, mod_format, None, safety_margin, N_ROADMs, N_OLAs
            
    # print(f"No wavelength meets the RSNR: {adjusted_snr_db} dB; path: {path}")
    # If we reach here, no wavelength met the SNR requirement
    return None, None, None, None, None, "insufficient_snr", None, None, None

def handle_traffic_demands_with_snr_modulation(
    traffic_demands, available_wavelengths, config, params, k,
    wavelength_assignment_strategy="first-fit", use_gaussian_modulation=True, modulation_formats=None, 
    gaussian_threshold=None, stop_on_wavelength_block=False, progressive_traffic=False, 
    max_blocking_probability=0.1, num_nodes=None, all_pairs_shortest_paths=None, G=None
):
    """
    Handle traffic demands for both static and progressive (dynamic) traffic scenarios.

    Parameters:
    - traffic_demands: List of demands [(source, target, paths, shortest_path_length)] (static scenario).
                      If None and `progressive_traffic` is True, demands are generated dynamically.
    - available_wavelengths: Dictionary of wavelength availability for links.
    - config: Configuration object containing system parameters.
    - params: Dictionary of system parameters.
    - k: Number of shortest paths to consider for each demand.
    - wavelength_assignment_strategy: Strategy for wavelength assignment.
    - use_gaussian_modulation: Boolean flag to toggle between Gaussian and real modulations.
    - modulation_formats: Dictionary of modulation formats and their SNR thresholds.
    - gaussian_threshold: Minimum SNR threshold for Gaussian modulation.
    - stop_on_wavelength_block: Boolean. If True, stops immediately upon detecting a wavelength-blocked demand.
    - progressive_traffic: Boolean. If True, dynamically generate traffic until `max_blocking_probability` is reached.
    - max_blocking_probability: Maximum blocking probability for progressive traffic mode.
    - num_nodes: Total number of nodes in the network (required for progressive traffic).
    - all_pairs_shortest_paths: Dictionary of k-shortest paths (required for progressive traffic).

    Returns:
    - successful_connections: List of successful demands with details.
    - blocked_connections: List of blocked demands with details.
    - modulation_stats: Dictionary with modulation usage statistics.
    - required_wavelengths: Number of wavelengths used in the network.
    - blocking_probs (optional): Blocking probabilities over time (progressive traffic only).
    - capacities (optional): Allocated capacities over time (progressive traffic only).
    """
    successful_connections = []
    blocked_connections = []
    modulation_stats = {mod: {"count": 0, "details": []} for mod in (["Gaussian"] if use_gaussian_modulation else modulation_formats)}
    required_wavelengths = set()
    used_wavelengths = set()

    # Track metrics for progressive traffic
    blocking_probs = []
    capacities = []
    total_demanded = 0  # Total traffic demands processed
    
    if progressive_traffic:
        # Check required parameters
        if num_nodes is None or all_pairs_shortest_paths is None:
            raise ValueError("For progressive traffic, `num_nodes` and `all_pairs_shortest_paths` must be provided.")

        while True:
            # Generate a new random demand
            demand1, demand2 = generate_random_demand(G, num_nodes, all_pairs_shortest_paths)
            traffic_demands = [demand1, demand2]  # Process one bidirectional pair
            total_demanded += len(traffic_demands)

            for demand in traffic_demands:
                source, target, paths, _ = demand
                wavelength_assigned = False
                snr_blocked = False

                # Try each of the k paths for the demand
                for path in paths[:k]:
                    result = assign_wavelength_with_snr_modulation_check(
                        path, available_wavelengths, config, params, 
                        assignment_strategy=wavelength_assignment_strategy,
                        use_gaussian_modulation=use_gaussian_modulation, 
                        modulation_formats=modulation_formats,
                        gaussian_threshold=gaussian_threshold
                    )

                    # Unpack the results
                    (assigned_wavelength, snr, adjusted_snr, capacity, modulation_used, 
                     block_reason, safety_margin, n_roadms, n_olas) = result

                    if assigned_wavelength is not None:
                        # Successful wavelength assignment
                        successful_connections.append({
                            "source": source,
                            "target": target,
                            "path": path,
                            "wavelength": assigned_wavelength,
                            "snr": snr,
                            "adjusted_snr": adjusted_snr,
                            "capacity": capacity,
                            "modulation": modulation_used,
                            "safety_margin": safety_margin,
                            "n_roadms": n_roadms,
                            "n_olas": n_olas
                        })
                        modulation_stats[modulation_used]["count"] += 1
                        modulation_stats[modulation_used]["details"].append({
                            "path": path,
                            "snr": snr,
                            "adjusted_snr": adjusted_snr,
                            "wavelength": assigned_wavelength,
                            "safety_margin": safety_margin,
                            "n_roadms": n_roadms,
                            "n_olas": n_olas
                        })
                        required_wavelengths.add(assigned_wavelength)
                        wavelength_assigned = True
                        break

                    elif block_reason == "insufficient_snr":
                        snr_blocked = True

                if not wavelength_assigned:
                    # If the demand couldn't be assigned, record the block reason
                    block_reason = "Insufficient Wavelengths" if not snr_blocked else "Insufficient SNR"
                    blocked_connections.append({
                        "source": source,
                        "target": target,
                        "path": paths[0],
                        "reason": block_reason
                    })

            # Calculate metrics for progressive traffic
            blocking_ratio = len(blocked_connections) / total_demanded
            total_capacity = sum(conn["capacity"] for conn in successful_connections)
            blocking_probs.append(blocking_ratio)
            capacities.append(total_capacity)

            # Print status
            print(f"Progressive Traffic: Total Demands = {total_demanded}, Blocking = {blocking_ratio:.2%}, Capacity = {total_capacity:.2f} Gbps")

            # Stop when blocking ratio exceeds the threshold
            if blocking_ratio >= max_blocking_probability:
                print(f"Stopping progressive traffic: Blocking ratio = {blocking_ratio:.2%} exceeded the threshold.")
                break
        
        
        return successful_connections, blocked_connections, modulation_stats, len(required_wavelengths), blocking_probs, capacities

    # Static Traffic Mode
    for demand in traffic_demands:
        source, target, paths, length = demand
        wavelength_assigned = False
        snr_blocked = False

        # print(f'Demand length: {length}')
        # Try each of the k paths for the demand
        for path in paths[:k]:
            result = assign_wavelength_with_snr_modulation_check(
                path, available_wavelengths, config, params, 
                assignment_strategy=wavelength_assignment_strategy,
                use_gaussian_modulation=use_gaussian_modulation, 
                modulation_formats=modulation_formats,
                gaussian_threshold=gaussian_threshold
            )

            # Unpack the results
            (assigned_wavelength, snr, adjusted_snr, capacity, modulation_used, 
             block_reason, safety_margin, n_roadms, n_olas) = result

            if assigned_wavelength is not None:
                successful_connections.append({
                    "source": source,
                    "target": target,
                    "path": path,
                    "wavelength": assigned_wavelength,
                    "snr": snr,
                    "adjusted_snr": adjusted_snr,
                    "capacity": capacity,
                    "modulation": modulation_used,
                    "safety_margin": safety_margin,
                    "n_roadms": n_roadms,
                    "n_olas": n_olas
                })
                modulation_stats[modulation_used]["count"] += 1
                modulation_stats[modulation_used]["details"].append({
                    "path": path,
                    "snr": snr,
                    "adjusted_snr": adjusted_snr,
                    "wavelength": assigned_wavelength,
                    "safety_margin": safety_margin,
                    "n_roadms": n_roadms,
                    "n_olas": n_olas
                })
                required_wavelengths.add(assigned_wavelength)
                used_wavelengths.add(assigned_wavelength)
                wavelength_assigned = True
                break

            elif block_reason == "insufficient_snr":
                snr_blocked = True

        if not wavelength_assigned:
            block_reason = "Insufficient Wavelengths" if not snr_blocked else "Insufficient SNR"
            blocked_connections.append({
                "source": source,
                "target": target,
                "path": paths[0],
                "reason": block_reason
            })

            if stop_on_wavelength_block and block_reason == "Insufficient Wavelengths":
                return successful_connections, blocked_connections, modulation_stats, len(required_wavelengths)

    return successful_connections, blocked_connections, modulation_stats, len(required_wavelengths), used_wavelengths


def handle_static_traffic(
    G, all_pairs_shortest_paths, precomputed_link_data, config
):
    """
    Handle the static traffic scenario, performing traffic demand ordering, wavelength assignment, 
    and capacity computation.

    Parameters:
    - G: Network graph representing the optical network.
    - all_pairs_shortest_paths: Dictionary of the shortest paths between all pairs of nodes in the network.
    - precomputed_link_data: Precomputed data for each band, including parameters like SNR, gains, spans, etc.
    - config: Configuration object containing system parameters, such as number of Monte Carlo iterations, band information, and traffic handling strategies.

    Returns:
    - static_results: List of results for each Monte Carlo iteration, including details like network capacity, average SNR, blocking ratio, and modulation statistics.
    - network_results: Detailed results from the first iteration (if NMC is 1).
    """

    static_results = []
    network_results = []  # For detailed per-network results

    # Get the last band name from the dictionary
    last_band_key_name = list(config.bands.keys())[-1]

    # Initialize total link saturation tracker
    total_link_saturation = {link: [] for link in G.edges}

    # Ensure both (u, v) and (v, u) are included explicitly for bidirectional edges
    for u, v in G.edges:
        total_link_saturation[(u, v)] = []
        total_link_saturation[(v, u)] = []

    for iteration in range(config.NMC):
        print(f"\nMonte Carlo Iteration {iteration + 1}/{config.NMC}")

        for band_name, params in config.bands.items():
            print(f"\n--- Simulating with {band_name} Band ---")
            link_data = precomputed_link_data[band_name]
            params = link_data["params"]
            params.update({
                "SNR_links": link_data["SNR_links"],
                "P_ch_opt": link_data["P_ch_opt_links"],
                "N_spans": link_data["N_spans_links"],
                "Gains": link_data["Gains_links"],
            })

            # -------------------------------------------------------------------------------- #
            # ---------------------- STEP 2: Order the Traffic Demands ----------------------- # 
            # -------------------------------------------------------------------------------- #
            print("\nSTEP 2: Order the Traffic Demands")
            sorting_start_time = time.perf_counter()
            sorted_traffic_demands = order_traffic_demands(G, all_pairs_shortest_paths, config.path_order_strategy)
            
            print(f'LONGEST SHORTEST-PATH: {sorted_traffic_demands[-2][0]} --> {sorted_traffic_demands[-2][1]} with LENGTH: {sorted_traffic_demands[-2][3]} km')
            longest_shortest_path = sorted_traffic_demands[-2][3]
            sorting_end_time = time.perf_counter()

            #for demand in sorted_traffic_demands:
             #   print(f'Demand: {demand[0]} -> {demand[1]} with Length: {demand[3]} and Path: {demand[2]}')

            # -------------------------------------------------------------------------------- #
            # -------- STEP 3, 4 and 5: Wavelength Assignment, Path SNR and Blocking --------- # 
            # -------------------------------------------------------------------------------- #
            print("\nSTEP 3, 4 and 5: Wavelength Assignment, Path SNR and Blocking")
            assignment_start_time = time.perf_counter()
            available_wavelengths = prepare_available_wavelengths(G, params["num_channels"])

            successful_connections, blocked_connections, modulation_stats, required_wavelengths, used_wavelengths = handle_traffic_demands_with_snr_modulation(
                sorted_traffic_demands,
                available_wavelengths,
                config,
                params,
                config.k_paths,
                wavelength_assignment_strategy=config.wavelength_assignment_strategy,
                use_gaussian_modulation=config.use_gaussian_modulation,
                modulation_formats=config.modulation_formats_with_thresholds,
                gaussian_threshold=config.gaussian_threshold,
                stop_on_wavelength_block=(band_name != config.bands[last_band_key_name]["band_name"]),
            )
            assignment_end_time = time.perf_counter()

            # Check if upgrade is needed or finalize results
            if band_name == config.bands[last_band_key_name]["band_name"] or all(blocked["reason"] != "Insufficient Wavelengths" for blocked in blocked_connections):
                print(f"No wavelength blocked demands in {band_name} Band. Finalizing Results.")
                
                # -------------------------------------------------------------------------------- #
                # ------------------------ STEP 6: Capacity Computation -------------------------- #
                # -------------------------------------------------------------------------------- #
                print("\nSTEP 6: Capacity Computation")

                # Compute and store results
                total_bit_rate = sum(conn["capacity"] for conn in successful_connections)  # Sum capacities
                num_allocated_lps = len(successful_connections)
                num_blocked_demands = len(blocked_connections)

                avg_channel_capacity = total_bit_rate / num_allocated_lps if num_allocated_lps > 0 else 0
                avg_snr = np.mean([conn["adjusted_snr"] for conn in successful_connections]) if successful_connections else 0
                blocking_ratio = num_blocked_demands / len(sorted_traffic_demands)
                network_capacity = total_bit_rate / 1e3  # Convert to Tbps

                # Blocked Demands by Reason
                snr_blocked_count = sum(1 for blocked in blocked_connections if blocked["reason"] == "Insufficient SNR")
                wavelength_blocked_count = sum(1 for blocked in blocked_connections if blocked["reason"] == "Insufficient Wavelengths")

                # Timing for individual steps
                sorting_time = sorting_end_time - sorting_start_time
                assignment_time = assignment_end_time - assignment_start_time

                # Calculate per-link saturation
                for link, wavelengths in available_wavelengths.items():
                    num_used_wavelengths  = sum(not available for available in wavelengths)
                    saturation = num_used_wavelengths  / params["num_channels"]
                    total_link_saturation[link].append(saturation)

                result = {
                    "band_used": band_name,
                    "sorting_time": sorting_time,
                    "assignment_time": assignment_time,
                    "average_snr": (10 * np.log10(avg_snr)),
                    "average_channel_capacity": avg_channel_capacity,
                    "network_capacity": network_capacity,
                    "blocking_ratio": blocking_ratio,
                    "traffic_demands": len(sorted_traffic_demands),
                    "blocked_demands": num_blocked_demands,
                    "blocked_due_to_snr": snr_blocked_count,  # Storing the reason for blocked demands
                    "blocked_due_to_wavelength": wavelength_blocked_count,  # Storing the reason for blocked demands
                    "required_wavelengths": required_wavelengths,  # Storing number of wavelengths required
                    "used_wavelengths": sorted(list(used_wavelengths)),  # Store sorted list of actually used wavelengths
                    "modulation_stats": modulation_stats,
                    "link_saturation": total_link_saturation, # Per iteration
                    "longest_shortest_path": longest_shortest_path,
                }
                static_results.append(result)

                if config.NMC == 1:
                    network_results.append(result)  # For single iteration, store detailed results
                break

    # Compute averaged link saturation over all iterations
    avg_link_saturation = {
        link: np.mean(loads) for link, loads in total_link_saturation.items()
    }

    if config.NMC > 1:
        aggregated_results = {
            "average_channel_capacity": np.mean([res["average_channel_capacity"] for res in static_results]),
            "blocking_ratio": np.mean([res["blocking_ratio"] for res in static_results]),
            "network_capacity": np.mean([res["network_capacity"] for res in static_results]),
            "average_snr": np.mean([res["average_snr"] for res in static_results]),
            "link_saturation": avg_link_saturation,  # Now added
        }
        return aggregated_results

    return network_results if config.NMC == 1 else static_results[0]  # Detailed results for NMC=1

def handle_progressive_traffic(
    G, all_pairs_shortest_paths, precomputed_link_data, config
):
    """
    Handle the progressive traffic scenario.
    
    Parameters:
        G (networkx.Graph): The network graph.
        all_pairs_shortest_paths (dict): Precomputed shortest paths between node pairs.
        precomputed_link_data (dict): Precomputed data for each link, including SNR, gains, etc.
        config (object): Configuration object containing simulation parameters such as bandwidth, number of channels, etc.
    
    Returns:
        dict: A dictionary containing the results for each band, including blocking probabilities, capacities, and link saturation.
    """
    mc_results = {band_name: {"blocking_probs": [], "capacities": [], "link_saturation": {}} for band_name, band in config.bands.items()}

    # Initialize link saturation trackers (bidirectional links)
    total_link_saturation = {band_name: {link: [] for link in G.edges} for band_name, band in config.bands.items()}

    # Ensure both (u, v) and (v, u) are included explicitly for bidirectional edges
    for band_name, band in config.bands.items():
        for u, v in G.edges:
            total_link_saturation[band_name][(u, v)] = []
            total_link_saturation[band_name][(v, u)] = []

    for iteration in range(config.NMC):
        print(f"\nMonte Carlo Iteration {iteration + 1}/{config.NMC}")

        for band_name, band in config.bands.items():
            print(f"\n--- Simulating with {band_name} Band ---")
            link_data = precomputed_link_data[band_name]
            params = link_data["params"]
            params.update({
                "SNR_links": link_data["SNR_links"],
                "P_ch_opt": link_data["P_ch_opt_links"],
                "N_spans": link_data["N_spans_links"],
                "Gains": link_data["Gains_links"],
            })

            available_wavelengths = prepare_available_wavelengths(G, params["num_channels"])

            # Simulate traffic until blocking threshold is reached
            _, _, _, _, blocking_probs, capacities = handle_traffic_demands_with_snr_modulation(
                None,
                available_wavelengths,
                config,
                params,
                config.k_paths,
                wavelength_assignment_strategy=config.wavelength_assignment_strategy,
                use_gaussian_modulation=config.use_gaussian_modulation,
                modulation_formats=config.modulation_formats_with_thresholds,
                gaussian_threshold=config.gaussian_threshold,
                progressive_traffic=True,
                max_blocking_probability=config.max_blocking_probability,
                num_nodes=len(G.nodes),
                all_pairs_shortest_paths=all_pairs_shortest_paths,
                G=G
            )

            # Calculate per-link saturation
            for link, wavelengths in available_wavelengths.items():
                used_wavelengths = sum(not available for available in wavelengths)
                saturation = used_wavelengths / params["num_channels"]
                total_link_saturation[band_name][link].append(saturation)

            mc_results[band_name]["blocking_probs"].append(blocking_probs)
            mc_results[band_name]["capacities"].append(capacities)

    # Average link saturation across all iterations
    for band_name, band in config.bands.items():
        mc_results[band_name]["link_saturation"] = {
            link: np.mean(saturation_list)
            for link, saturation_list in total_link_saturation[band_name].items()
        }

    return mc_results