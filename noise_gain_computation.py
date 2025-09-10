"""
File: noise_gain_computation.py

Author: Tomás Maia
Date: 2025-02-21

Description:
This module contains various functions for calculating ISRS gain/loss, ASE noise, and amplifier gain in optical communication systems. The functions handle both contiguous and non-contiguous frequency bands and account for fiber attenuation and multiplexing losses. They are essential for modeling the physical layer of optical networks, including the effects of Raman scattering and amplifier noise.

Functions:
- compute_isrs_gain_loss: Computes ISRS gain or loss for each channel over a given span length.
- compute_roadm_noise: Computes the ASE noise contribution from a single ROADM node for all channels.
- compute_ase_noise_isrs: Computes the ASE noise per channel for spans, using the actual channel frequencies.
- compute_post_amplifier_gain: Computes the post-amplifier gain for nodes.
- compute_line_amplifier_gain: Computes the in-line amplifier gain per frequency component, considering fiber attenuation and ISRS.

Dependencies:
- numpy

"""

import numpy as np

def compute_isrs_gain_loss(Ptot, channels, Cr, Leff, BW_WDM, rel_channel_freq, 
                          non_contiguous=False, rel_freq_band1=None, rel_freq_band2=None):
    """
    Computes the ISRS gain or loss for each channel over a given span length.

    Parameters:
    - Ptot: Total power launched into the fiber (in Watts).
    - channels: Number of WDM channels.
    - Cr: Raman gain slope (W^-1·m^-1·Hz^-1).
    - Leff: Effective fiber length (in meters).
    - BW_WDM: WDM bandwidth (in Hz).
    - rel_channel_freq: Center frequencies of WDM channels (relative to reference frequency).
    - non_contiguous: Flag indicating if the spectrum is non-contiguous (default: False).
    - rel_freq_band1: Tuple (min_freq, max_freq) for Band 1, required if non_contiguous is True.
    - rel_freq_band2: Tuple (min_freq, max_freq) for Band 2, required if non_contiguous is True.

    Returns:
    - ISRS_gain: ISRS-induced gain or loss for each channel (in linear scale).
    """
    if non_contiguous:
        if rel_freq_band1 is None or rel_freq_band2 is None:
            raise ValueError("For non-contiguous bands, 'rel_freq_band1' and 'rel_freq_band2' must be provided.")

        # Extract min and max relative frequencies for each band
        min_freq_band1, max_freq_band1 = rel_freq_band1
        min_freq_band2, max_freq_band2 = rel_freq_band2

        # Compute the denominator considering the non-contiguous spectrum
        denominator = (
            np.exp(Cr * Ptot * Leff * max_freq_band1) - np.exp(Cr * Ptot * Leff * min_freq_band1) +
            np.exp(Cr * Ptot * Leff * max_freq_band2) - np.exp(Cr * Ptot * Leff * min_freq_band2)
        )
    else:
        # Contiguous spectrum: use total WDM bandwidth for the denominator
        denominator = (
            np.exp(Cr * Ptot * Leff * BW_WDM / 2) - np.exp(-Cr * Ptot * Leff * BW_WDM / 2)
        )

    # Compute ISRS gain/loss for each channel
    ISRS_gain = (
        Cr * Ptot * Leff * BW_WDM * np.exp(-Cr * Ptot * Leff * rel_channel_freq)
    ) / denominator

    return ISRS_gain

def compute_roadm_noise(post_amp_gain, fn, Bch, h, nu_ch):
    """
    Compute the ASE noise contribution from a single ROADM node for all channels.
    
    Parameters:
    - post_amp_gain: Gain of the post-amplifier (in dB) for the ROADM.
    - fn: Noise figure of the EDFA compensating the ROADM loss (in linear scale).
    - Bch: Channel bandwidth (in Hz).
    - h: Planck's constant (in J·s).
    - nu_ch: Array of channel frequencies (in Hz).
    
    Returns:
    - ASE_noise: Array of ASE noise contributions for each channel (in Watts).
    """
    G_linear = 10 ** (post_amp_gain / 10)  # Convert gain from dB to linear scale
    ASE_noise = h * nu_ch * fn * (G_linear - 1) * Bch
    return ASE_noise

def compute_ase_noise_isrs(Bch, Ns, h, nu_ch, fn, amplifier_gain):
    """
    Compute the ASE noise per channel for spans, using the actual channel frequencies.
    
    Parameters:
    - Bch: Channel bandwidth (in Hz).
    - Ns: Number of spans.
    - h: Planck's constant (in J·s).
    - nu_ch: Array of channel frequencies (in Hz).
    - fn: Noise figure (linear scale).
    - amplifier_gain: Gain per channel considering ISRS and fiber attenuation (in dB).

    Returns:
    - ASE_noise_ISRS: Array of ASE noise contributions for each channel (in Watts).
    """
    G_linear = 10 ** (amplifier_gain / 10)  # Convert amplifier gain from dB to linear scale
    ASE_noise_ISRS = Ns * h * nu_ch * fn * (G_linear - 1) * Bch # Compute ASE noise per channel
    return ASE_noise_ISRS

def compute_post_amplifier_gain(node_type, next_link_power, current_link_power, mux_demux_loss):
    """
    Compute the post-amplifier gain for nodes.
    
    Parameters:
    - node_type: Type of node ("add", "drop", "intermediate").
    - next_link_power: LOGON optimal power for the next link (in dBm).
    - current_link_power: Power of the current link (in dBm).
    - mux_demux_loss:  Insertion losses of the required band multiplexer and demultiplexer at every amplification stage
    
    Returns:
    - Post-amplifier gain (in dB).
    """
    base_loss = mux_demux_loss  # Mux + Demux losses (2 dB + 1 dB)
    if node_type == "intermediate":
        additional_loss = 18
        gain_adjustment = next_link_power - current_link_power
    elif node_type in ["add", "drop"]:
        additional_loss = 15
        gain_adjustment = 0  # Add/drop nodes only compensate their own losses
    else:
        raise ValueError("Invalid node type. Choose from 'add', 'drop', or 'intermediate'.")

    # print(f'Current Link Power: {current_link_power} dBm\nNext Link Power: {next_link_power} dBm\n Gain Adjustment: {gain_adjustment} dB')
    
    return base_loss + additional_loss + gain_adjustment

def compute_line_amplifier_gain(alpha_dB, Ls, Cr, Ptot, Leff, BW_WDM, channels, mux_demux_loss, rel_channel_freq, non_contiguous=False, rel_freq_band1=None, rel_freq_band2=None):
    """
    Compute the in-line amplifier gain per frequency component, considering fiber attenuation and ISRS.
    
    Parameters:
    - alpha_dB: Attenuation coefficient (in dB/km).
    - Ls: Span length in km.
    - Cr: Raman gain slope (in W^-1·m^-1·Hz^-1).
    - Ptot: Total power in the fiber (in Watts).
    - Leff: Effective length (in meters).
    - BW_WDM: Total WDM bandwidth (in Hz).
    - channels: Number of channels.
    - mux_demux_loss: Insertion losses of the required band multiplexer and demultiplexer at every amplification stage.
    - rel_channel_freq: Center frequencies of WDM channels (relative to reference frequency).
    - non_contiguous: Boolean flag for non-contiguous spectrum.
    - rel_freq_band1: Tuple of (min_freq, max_freq) for Band 1 (required if non-contiguous).
    - rel_freq_band2: Tuple of (min_freq, max_freq) for Band 2 (required if non-contiguous).

    Returns:
    - amplifier_gain: Gain per channel considering ISRS, fiber attenuation, and mux+demux losses.
    """

    # Compute ISRS gain/loss for each channel, considering non-contiguous bands
    ISRS_gain = compute_isrs_gain_loss(
        Ptot, channels, Cr, Leff, BW_WDM, rel_channel_freq,
        non_contiguous=non_contiguous,
        rel_freq_band1=rel_freq_band1,
        rel_freq_band2=rel_freq_band2
    )

    # Fiber attenuation (in dB)
    fiber_loss = alpha_dB * Ls

    # Amplifier gain compensating for attenuation, ISRS, and mux+demux losses
    amplifier_gain = fiber_loss + mux_demux_loss - 10 * np.log10(ISRS_gain)

    return amplifier_gain