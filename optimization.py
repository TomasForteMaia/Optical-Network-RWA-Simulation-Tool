"""
optimization.py

Author: Tomás Maia 96340 MEEC
Date: 10/02/2025

Description:
This module performs power optimization for optical network links, considering ISRS effects
and dynamic amplifier gain calculations. It also computes SNR for different modulation formats.

Main functions:
1. perform_power_optimization_with_isrs - Computes optimal channel power considering ISRS.
2. optimize_links - Optimizes power and computes SNR for each link in a network graph.
3. precompute_links - Precomputes link optimizations for all defined bands.

"""

import numpy as np
from noise_gain_computation import compute_line_amplifier_gain
from calculate_snr import compute_link_snr

def perform_power_optimization_with_isrs(length_km, fn_value, config, params, non_contiguous=False, rel_freq_band1=None, rel_freq_band2=None):
    """
    Computes the optimal channel power considering ISRS and amplifier gain.
    
    Parameters:
    - length_km (float): Length of the optical link (km).
    - fn_value (float): Noise figure parameter for ASE noise calculation.
    - config (object): Configuration object containing system parameters.
    - params (dict): Dictionary of physical layer parameters (e.g., reference frequency, beta2, bandwidth).
    - non_contiguous (bool, optional): True if the bands are non-contiguous, False otherwise.
    - rel_freq_band1 (tuple, optional): Min/max relative frequencies for first band in non-contiguous case.
    - rel_freq_band2 (tuple, optional): Min/max relative frequencies for second band in non-contiguous case.
    
    Returns:
    - P_ch_opt_dBm (float): Optimized channel power in dBm.
    - Ns (int): Number of spans in the link.
    - amplifier_gain (float): Amplifier gain per channel in dB.
    """

    # Calculate the number of spans required for the given link length
    Ns = int(np.ceil(length_km / config.Ls_max))  # Number of spans
    Ls = length_km / Ns  # Actual span length in km

    # Compute the effective fiber length considering attenuation
    alpha_N = config.alpha_dB / (10 * np.log10(np.e))  # Convert dB/km to natural scale
    Leff = (1 - np.exp(- alpha_N * Ls)) / (alpha_N)  # Effective length in km
    Leff = Leff * 1e3  # Convert to meters

    # Compute ASE noise power spectral density for the given link
    As_dB = config.alpha_dB * Ls  # Fiber attenuation in dB per span
    As = 10**(As_dB / 10)  # Convert dB to linear scale
    ASE_noise = Ns * config.h * params['ref_frequency'] * fn_value * (As - 1) * config.Bch  # ASE noise accumulation

    # Compute the nonlinear interference (NLI) coefficient
    mu_n = ((2 / 3) ** 3) * Ns * (config.gamma ** 2) * Leff * np.log((np.pi ** 2) * abs(params['beta2']) * Leff * (params['bandwidth'] ** 2)) / (np.pi * (config.Bch ** 3) * abs(params['beta2']))

    # Compute the optimal channel power without ISRS
    P_ch_opt_linear = np.power(ASE_noise / (2 * mu_n * config.Bch), 1/3)
    P_ch_opt_dBm = 10 * np.log10(P_ch_opt_linear) + 30  # Convert to dBm

    # Compute the amplifier gain dynamically considering ISRS effects
    amplifier_gain = compute_line_amplifier_gain(config.alpha_dB, Ls, config.Cr, Ptot=P_ch_opt_linear * params['num_channels'], Leff=Leff, BW_WDM=params['bandwidth'], channels=params['num_channels'], mux_demux_loss=params['mux_demux_loss'], rel_channel_freq=params['relative_freqs'], non_contiguous=non_contiguous, rel_freq_band1=rel_freq_band1, rel_freq_band2=rel_freq_band2)

    return P_ch_opt_dBm, Ns, amplifier_gain

def optimize_links(graph, config, params):
    """
    Optimizes channel power and computes SNR for all links in the network graph.
    
    Parameters:
    - graph (networkx.Graph): Graph representing the optical network.
    - config (object): Configuration containing network and system parameters.
    - params (dict): Dictionary containing relevant physical layer parameters.
    
    Returns:
    - SNR_links (dict): SNR values for each link and modulation format.
    - P_ch_opt_links (dict): Optimized channel power values for each link.
    - N_spans_links (dict): Number of spans for each link.
    - Gains_links (dict): Amplifier gain values for each link.
    """
    P_ch_opt_links = {}
    N_spans_links = {}
    Gains_links = {}
    SNR_links = {"Gaussian": {}}

    if not config.use_gaussian_modulation:
        for mod_format in config.modulation_formats:
            SNR_links[mod_format] = {}

    for u, v, data in graph.edges(data=True):
        length_km = data["weight"]
        link = (u, v)
        inverse_link = (v, u)

        fn_value = params['fn'] if isinstance(params['fn'], float) else params['fn'][1]  # Use L-band fn for multi-band cases

        # Determine if the band is contiguous or non-contiguous
        band_name = params['band_name']
        band = config.all_bands[band_name]
        if " + " in band_name:
            band1, band2 = band_name.split(" + ")
            num_channels_band1 = config.all_bands[band1]["num_channels"]
            num_channels_band2 = config.all_bands[band2]["num_channels"]
            
            rel_freq_band1 = (
                np.nanmin(params['relative_freqs'][:num_channels_band2]),
                np.nanmax(params['relative_freqs'][:num_channels_band2])
            ) if not params['contiguous'] else None
            rel_freq_band2 = (
                np.nanmin(params['relative_freqs'][num_channels_band2:]),
                np.nanmax(params['relative_freqs'][num_channels_band2:])
            ) if not params['contiguous'] else None
        else:
            rel_freq_band1 = None
            rel_freq_band2 = None
            
        # Perform power optimization
        P_opt, Ns, Gin_dB = perform_power_optimization_with_isrs(
            length_km, fn_value, config, params, non_contiguous=not params['contiguous'],
            rel_freq_band1=rel_freq_band1,
            rel_freq_band2=rel_freq_band2
        )

        # Store results
        P_ch_opt_links[link] = P_opt
        N_spans_links[link] = Ns
        Gains_links[link] = Gin_dB

        P_ch_opt_links[inverse_link] = P_opt
        N_spans_links[inverse_link] = Ns
        Gains_links[inverse_link] = Gin_dB

        # Compute SNR
        if config.use_gaussian_modulation:
            SNR_links["Gaussian"][link] = compute_link_snr(length_km, P_opt, Ns, Gin_dB, config, params, include_nli=True, gaussian=True)
            SNR_links["Gaussian"][inverse_link] = SNR_links["Gaussian"][link]
        else:
            for mod_format in config.modulation_formats:
                SNR_links[mod_format][link] = compute_link_snr(
                    length_km, P_opt, Ns, Gin_dB, config, params, include_nli=True, gaussian=False, modulation=mod_format
                )
                SNR_links[mod_format][inverse_link] = SNR_links[mod_format][link]

    # Update parameters
    params['P_ch_opt'] = P_ch_opt_links
    params['N_spans'] = N_spans_links
    params['Gains'] = Gains_links

    return SNR_links, P_ch_opt_links, N_spans_links, Gains_links

def precompute_links(G, config):
    """
    Precomputes optimized power and SNR values for all bands.
    
    Parameters:
    - G (networkx.Graph): Graph representation of the optical network.
    - config (object): Configuration object containing system parameters.
    
    Returns:
    - precomputed_link_data (dict): Dictionary containing optimization results for all bands.
    """
    precomputed_link_data = {}
    for band_name, params in config.bands.items():
        print(f"Precomputing link optimization for {band_name} band...")

        params["add_drop"] = config.add_drop
        params["OH"] = config.OH
        SNR_links, P_ch_opt_links, N_spans_links, Gains_links = optimize_links(
            G, config, params
        )
        precomputed_link_data[band_name] = {
            "SNR_links": SNR_links,
            "P_ch_opt_links": P_ch_opt_links,
            "N_spans_links": N_spans_links,
            "Gains_links": Gains_links,
            "params": params,
        }
    return precomputed_link_data