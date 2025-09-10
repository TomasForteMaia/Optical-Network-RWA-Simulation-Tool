"""
Module: Band Definition
Author: Tomás Maia 96340 MEEC
Date: 10/02/2025

This module defines optical transmission bands and computes relevant parameters such as frequencies, 
dispersion coefficients, and noise figures. It supports both contiguous and non-contiguous multi-band systems.
"""

import numpy as np

def compute_channel_frequencies(center_freq, num_channels, spacing):
    """
    Compute WDM channel frequencies based on center frequency and spacing.

    Parameters:
    - center_freq: Reference frequency (Hz).
    - num_channels: Number of channels in the system.
    - spacing: Channel spacing (Hz).

    Returns:
    - Array of channel frequencies (Hz).
    """
    return center_freq + (np.arange(num_channels) - (num_channels - 1) / 2) * spacing
    
def compute_relative_frequencies(center_freq_c, num_channels_c, center_freq_l, num_channels_l, spacing, contiguous=True):
    """
    Compute relative frequencies for WDM channels based on center frequencies and spacing.

    Parameters:
    - center_freq_c: Center frequency of the first band (Hz).
    - num_channels_c: Number of channels in the first band.
    - center_freq_l: Center frequency of the second band (Hz).
    - num_channels_l: Number of channels in the second band.
    - spacing: Channel spacing (Hz).
    - contiguous: Boolean indicating if the bands are contiguous.

    Returns:
    - relative_frequencies: Array of relative frequencies (Hz).
    """
    if contiguous:
        # If the bands are contiguous, generate a single frequency array.
        total_channels = num_channels_c + num_channels_l
        return (np.arange(total_channels) - (total_channels - 1) / 2) * spacing
    else:
        # Compute absolute frequencies for each band.
        absolute_freqs_c = center_freq_c + (np.arange(num_channels_c) - (num_channels_c - 1) / 2) * spacing
        absolute_freqs_l = center_freq_l + (np.arange(num_channels_l) - (num_channels_l - 1) / 2) * spacing
        
        # Combine absolute frequencies of both bands.
        absolute_freqs_combined = np.concatenate([absolute_freqs_l, absolute_freqs_c])
        
        # Define the reference frequency as the midpoint of the min and max frequencies.
        reference_frequency = (min(absolute_freqs_combined) + max(absolute_freqs_combined)) / 2
        
        # Compute relative frequencies by subtracting the reference frequency.
        return absolute_freqs_combined - reference_frequency

def define_bands(contiguous=True, spacing=75e9, Bch=64e9, D=17e-12 / 1e-9 / 1e3):
    """
    Define the optical transmission bands and compute relevant parameters.

    Parameters:
    - contiguous: Boolean flag indicating if the bands should be contiguous.
    - spacing: Channel spacing in Hz.
    - Bch: Channel bandwidth in Hz.
    - D: Dispersion coefficient (in s/m^2).

    Returns:
    - Dictionary containing parameters for each defined optical band.
    """
    c = 2.99792458e8  # Speed of light in meters/second

    # Define function for common band parameters
    def compute_band_parameters(start_wavelength, bandwidth, fn_dB, mux_demux_loss, band_name):
        end_frequency = c / start_wavelength
        start_frequency = end_frequency - bandwidth
        end_wavelength = c / start_frequency
        ref_frequency = (start_frequency + end_frequency) / 2
        ref_lambda = c / ref_frequency
        num_channels = int(bandwidth / spacing)

        freqs = compute_channel_frequencies(ref_frequency, num_channels, spacing)

        rel_freqs = (np.arange(num_channels) - (num_channels - 1) / 2) * spacing
        beta2 = -D * ref_lambda**2 / (2 * np.pi * c)
        fn_linear = 10**(fn_dB / 10)

        return {
            "start_wavelength": start_wavelength,
            "end_wavelength": end_wavelength,
            "ref_lambda": ref_lambda,
            "start_frequency": start_frequency,
            "end_frequency": end_frequency,
            "ref_frequency": ref_frequency,
            "bandwidth": bandwidth,
            "num_channels": num_channels,
            "freqs": freqs,
            "relative_freqs": rel_freqs,
            "beta2": beta2,
            "fn": fn_linear,
            "mux_demux_loss": mux_demux_loss,
            "contiguous": True,  # Single-band systems are always contiguous
            "band_name": band_name
        }

    # Band definitions
    bands = {
        "C": compute_band_parameters(1529e-9, 4.8e12, 4.25, 0, "C"),
        "L": compute_band_parameters(1570e-9, 4.8e12, 4.69, 0, "L"),
        "Super C": compute_band_parameters(1524e-9, 6e12, 4.65, 0, "Super C"),
        "Super L": compute_band_parameters(1575e-9, 6e12, 5.09, 0, "Super L")
    }
    
    # Multi-band definitions
    def compute_multiband_parameters(band1, band2, contiguous):

        if contiguous:
            # start_frequency = min(bands[band1]["start_frequency"], bands[band2]["start_frequency"])
            # end_frequency = start_frequency + bands[band1]["bandwidth"] + bands[band2]["bandwidth"]
            # start_wavelength = c / end_frequency
            # end_wavelength = c / start_frequency

            # bandwidth = end_frequency - start_frequency  # Continuous spectrum
            # ref_frequency = (start_frequency + end_frequency) / 2

            start_wavelength = min(bands[band1]["start_wavelength"], bands[band2]["start_wavelength"])
            end_frequency = c / start_wavelength
            start_frequency = end_frequency - bands[band1]["bandwidth"] - bands[band2]["bandwidth"]
            end_wavelength = c / start_frequency

            bandwidth = end_frequency - start_frequency  # Continuous spectrum
            ref_lambda = (start_wavelength + end_wavelength) / 2
            ref_frequency = c / ref_lambda

            num_channels = int(bandwidth / spacing)

        else:
            start_frequency = min(bands[band1]["start_frequency"], bands[band2]["start_frequency"])
            end_frequency = max(bands[band1]["end_frequency"], bands[band2]["end_frequency"])

            start_wavelength = c / end_frequency
            end_wavelength = c / start_frequency

            bandwidth = bands[band1]["bandwidth"] + bands[band2]["bandwidth"]  # Sum of individual bandwidths
            ref_frequency = (start_frequency + end_frequency) / 2
            ref_lambda = c / ref_frequency
            
            num_channels = bands[band1]["num_channels"] + bands[band2]["num_channels"]

        if contiguous:
            # Compute actual channel frequencies for contiguous case
            freqs = compute_channel_frequencies(ref_frequency, num_channels, spacing)
            rel_freqs = (np.arange(num_channels) - (num_channels - 1) / 2) * spacing
        else:
            # Compute actual channel frequencies separately for Band 1 and Band 2
            freqs_band1 = compute_channel_frequencies(bands[band1]["ref_frequency"], bands[band1]["num_channels"], spacing)
            freqs_band2 = compute_channel_frequencies(bands[band2]["ref_frequency"], bands[band2]["num_channels"], spacing)
            freqs = np.concatenate([freqs_band2, freqs_band1])

            # Compute relative frequencies based on the multi-band reference
            rel_freqs = compute_relative_frequencies(
                bands[band1]["ref_frequency"], bands[band1]["num_channels"],
                bands[band2]["ref_frequency"], bands[band2]["num_channels"],
                spacing, contiguous
            )

        fn = [bands[band1]["fn"], bands[band2]["fn"]]
        beta2 = -D * ref_lambda**2 / (2 * np.pi * c)
        mux_demux_loss = 2 if band1 == "C" else 2  # C+L = 2 dB, Super C + Super L = 2 dB

        return {
            "start_wavelength": start_wavelength,
            "end_wavelength": end_wavelength,
            "ref_lambda": ref_lambda,
            "start_frequency": start_frequency,
            "end_frequency": end_frequency,
            "ref_frequency": ref_frequency,
            "bandwidth": bandwidth,
            "num_channels": num_channels,
            "freqs": freqs,
            "relative_freqs": rel_freqs,
            "beta2": beta2,
            "fn": fn,
            "mux_demux_loss": mux_demux_loss,
            "contiguous": contiguous,  # Store whether the band is contiguous or not
            "band_name": f"{band1} + {band2}"
        }

    bands["C + L"] = compute_multiband_parameters("C", "L", contiguous)
    bands["Super C + Super L"] = compute_multiband_parameters("Super C", "Super L", contiguous)

    return bands