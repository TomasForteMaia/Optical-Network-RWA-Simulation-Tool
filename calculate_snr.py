"""
calculate_snr.py

This module defines functions for calculating the Signal-to-Noise Ratio (SNR) in optical networks, including computing link SNR and calculating SNR for a path.

Author: Tomás Maia 96340 MEEC
Date: 10/02/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from ISRSGNmodelNew import ISRSGNmodel, ISRSGNmodelCorrected, ISRSGNmodelCorrectedFinal  # imports ISRS GN model
from noise_gain_computation import compute_ase_noise_isrs, compute_post_amplifier_gain, compute_roadm_noise
from path_computation import path_distance

def compute_link_snr(length_km, P_opt, N_spans, gains_dB, config, params, include_nli=True, gaussian=True, modulation=None):
    """
    Compute the SNR for a given link.

    Parameters:
    - length_km: Length of the link in kilometers.
    - P_opt: Optimized channel power in dBm.
    - N_spans: Number of spans in the link.
    - gains_dB: Amplifier gains in dB.
    - config: Configuration object containing system parameters.
    - params: Dictionary of system parameters.
    - include_nli: Boolean to include Non-Linear Interference (NLI) noise.
    - gaussian: Boolean to use Gaussian modulation.
    - modulation: Modulation format (if not using Gaussian modulation).

    Returns:
    - SNR: Signal-to-Noise Ratio for the link.
    """
    middle_index = params["num_channels"] // 2

    Pch_linear = 10 ** (P_opt / 10) * 0.001 * np.ones([params['num_channels'], N_spans])

    P = {
        'fi': np.repeat(params['relative_freqs'].reshape(-1, 1), N_spans, axis=1),
        'n': N_spans,
        'Bch': np.tile(config.Bch, [params['num_channels'], N_spans]),
        'RefLambda': params['ref_lambda'],
        'D': config.D * np.ones(N_spans),
        'S': config.S * np.ones(N_spans),
        'Att': config.alpha_dB / 4.343 / 1e3 * np.ones([params['num_channels'], N_spans]),
        'Cr': config.Cr * np.ones([params['num_channels'], N_spans]),
        'gamma': config.gamma * np.ones(N_spans),
        'Length': length_km * 1e3 / N_spans * np.ones(N_spans),
        'coherent': 0,
    }
    P['Att_bar'] = P['Att']

    band_name = params['band_name']
    if " + " in band_name:
        # Handle multi-band case
        if params['contiguous']:
            middle_index = params['num_channels'] // 2
            ASE_band1 = compute_ase_noise_isrs(
                config.Bch, N_spans, config.h, 
                params['freqs'][middle_index:], 
                params['fn'][0], 
                gains_dB[middle_index:]
            )
            ASE_band2 = compute_ase_noise_isrs(
                config.Bch, N_spans, config.h, 
                params['freqs'][:middle_index], 
                params['fn'][1], 
                gains_dB[:middle_index]
            )
            ASE = np.concatenate([ASE_band2, ASE_band1])
        else:
            band1, band2 = band_name.split(" + ")
            num_channels_band1 = config.all_bands[band1]["num_channels"]
            num_channels_band2 = config.all_bands[band2]["num_channels"]

            ASE_band1 = compute_ase_noise_isrs(
                config.Bch, N_spans, config.h, 
                params['freqs'][num_channels_band2:], 
                params['fn'][0], 
                gains_dB[num_channels_band2:]
            )
            ASE_band2 = compute_ase_noise_isrs(
                config.Bch, N_spans, config.h, 
                params['freqs'][:num_channels_band2], 
                params['fn'][1], 
                gains_dB[:num_channels_band2]
            )
            ASE = np.concatenate([ASE_band2, ASE_band1])
    else:
        ASE = compute_ase_noise_isrs(config.Bch, N_spans, config.h, params['freqs'], params['fn'], gains_dB)

    NLI = 0
    if include_nli:
        if gaussian:
            NLI, eta = ISRSGNmodel(Pch=Pch_linear, **P)
        else:
            NLI, eta = ISRSGNmodelCorrected(Pch=Pch_linear, modulation_format=modulation, **P)

    SNR = Pch_linear[0, 0] / (ASE + NLI)
    return SNR


def calculate_snr_for_path(path, config, params, precomputed_SNR_links, modulation="Gaussian"):
    """
    Calculate SNR for a given path based on the new LOGON network model.

    Parameters:
    - path: List of nodes in the path.
    - config: Configuration object containing system parameters.
    - params: Dictionary of system parameters.
    - precomputed_SNR_links: Dictionary of precomputed SNRs for each link.
    - modulation: Modulation format (default is "Gaussian").

    Returns:
    - total_SNR: Array of SNR values for all wavelengths in the path.
    """
    # Extract parameters
    RefFrequency = params['ref_frequency']
    RefLambda = params['ref_lambda']
    channel_frequencies = params['freqs']
    rel_channel_frequencies = params['relative_freqs']
    num_channels = params['num_channels']
    fn = params['fn']
    N_spans = params['N_spans']
    P_ch_opt = params['P_ch_opt']
    Gains = params['Gains']
    mux_demux_loss = params['mux_demux_loss']
    add_drop = params['add_drop']
    band_name = params['band_name']

    # Initialize total inverse SNR
    total_inverse_SNR = 0

    if add_drop:
        link_power = P_ch_opt[(path[0], path[1])]
        post_amp_gain_add = compute_post_amplifier_gain("add", next_link_power=link_power, current_link_power=link_power, mux_demux_loss=mux_demux_loss)
        
        if " + " in band_name:
            if params["contiguous"]:
                band1, band2 = band_name.split(" + ")
                middle_index = num_channels // 2
                ASE_band1 = compute_roadm_noise(post_amp_gain_add, config.all_bands[band1]["fn"], config.Bch, config.h, channel_frequencies[middle_index:])
                ASE_band2 = compute_roadm_noise(post_amp_gain_add, config.all_bands[band2]["fn"], config.Bch, config.h, channel_frequencies[:middle_index])
                ASE_add = np.concatenate([ASE_band2, ASE_band1])
            else:
                band1, band2 = band_name.split(" + ")
                num_channels_band2 = config.all_bands[band2]["num_channels"]

                ASE_band1 = compute_roadm_noise(post_amp_gain_add, config.all_bands[band1]["fn"], config.Bch, config.h, channel_frequencies[num_channels_band2:])
                ASE_band2 = compute_roadm_noise(post_amp_gain_add, config.all_bands[band2]["fn"], config.Bch, config.h, channel_frequencies[:num_channels_band2])
                ASE_add = np.concatenate([ASE_band2, ASE_band1])
        else:
            ASE_add = compute_roadm_noise(post_amp_gain_add, fn, config.Bch, config.h, channel_frequencies)

        total_inverse_SNR += ASE_add / (10 ** (link_power / 10) * 0.001)

    # Iterate through links in the path
    for i in range(len(path) - 1):
        link = (path[i], path[i + 1])

        current_link_power_dBm = P_ch_opt[link]

        # Retrieve the precomputed SNR for the link
        SNR_link = precomputed_SNR_links[link]
        total_inverse_SNR += 1 / SNR_link  # Add the link's inverse SNR

        # Add intermediate ROADM contribution (if not the last link)
        if i < len(path) - 2:
            next_link = (path[i + 1], path[i + 2])
            next_link_power_dBm = P_ch_opt[next_link]
            post_amp_gain = compute_post_amplifier_gain("intermediate", next_link_power_dBm, current_link_power_dBm, mux_demux_loss)
            
            if " + " in band_name:
                if params["contiguous"]:
                    middle_index = num_channels // 2
                    ASE_band1 = compute_roadm_noise(post_amp_gain, config.all_bands[band1]["fn"], config.Bch, config.h, channel_frequencies[middle_index:])
                    ASE_band2 = compute_roadm_noise(post_amp_gain, config.all_bands[band2]["fn"], config.Bch, config.h, channel_frequencies[:middle_index])
                    ASE_node = np.concatenate([ASE_band2, ASE_band1])
                else:
                    band1, band2 = band_name.split(" + ")
                    num_channels_band2 = config.all_bands[band2]["num_channels"]

                    ASE_band1 = compute_roadm_noise(post_amp_gain, config.all_bands[band1]["fn"], config.Bch, config.h, channel_frequencies[num_channels_band2:])
                    ASE_band2 = compute_roadm_noise(post_amp_gain, config.all_bands[band2]["fn"], config.Bch, config.h, channel_frequencies[:num_channels_band2])
                    ASE_node = np.concatenate([ASE_band2, ASE_band1])
            else:
                ASE_node = compute_roadm_noise(post_amp_gain, fn, config.Bch, config.h, channel_frequencies)
            
            total_inverse_SNR += ASE_node / (10 ** (next_link_power_dBm / 10) * 0.001)

    # Add-drop ROADM noise at the end of the path if applicable
    if add_drop:
        post_amp_gain_drop = compute_post_amplifier_gain("drop", next_link_power=None, current_link_power=P_ch_opt[(path[len(path) - 2], path[len(path) - 1])], mux_demux_loss=mux_demux_loss)
        if " + " in band_name:
            if params["contiguous"]:
                middle_index = num_channels // 2
                ASE_band1 = compute_roadm_noise(post_amp_gain_drop, config.all_bands[band1]["fn"], config.Bch, config.h, channel_frequencies[middle_index:])
                ASE_band2 = compute_roadm_noise(post_amp_gain_drop, config.all_bands[band2]["fn"], config.Bch, config.h, channel_frequencies[:middle_index])
                ASE_drop = np.concatenate([ASE_band2, ASE_band1])
            else:
                band1, band2 = band_name.split(" + ")
                num_channels_band2 = config.all_bands[band2]["num_channels"]

                ASE_band1 = compute_roadm_noise(post_amp_gain_drop, config.all_bands[band1]["fn"], config.Bch, config.h, channel_frequencies[num_channels_band2:])
                ASE_band2 = compute_roadm_noise(post_amp_gain_drop, config.all_bands[band2]["fn"], config.Bch, config.h, channel_frequencies[:num_channels_band2])
                ASE_drop = np.concatenate([ASE_band2, ASE_band1])
        else:
            ASE_drop = compute_roadm_noise(post_amp_gain_drop, fn, config.Bch, config.h, channel_frequencies)
        total_inverse_SNR += ASE_drop / (10 ** (P_ch_opt[(path[len(path) - 2], path[len(path) - 1])] / 10) * 0.001)
        
    # Final total SNR
    total_SNR = 1 / total_inverse_SNR

    return total_SNR