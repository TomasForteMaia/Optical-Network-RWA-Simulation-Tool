"""
modulation_ber_computation.py

This module provides functions to compute Bit-Error-Rate (BER) and Signal-to-Noise Ratio (SNR) 
for different modulation formats, including Quadrature Amplitude Modulation (QAM) and 
Phase-Shift Keying (PSK). It also includes functions to determine SNR thresholds for a target BER 
and compute the Gaussian modulation threshold based on real modulations.

Author: Tomás Maia 96340 MEEC
Date: 10/02/2025
"""

import numpy as np
from scipy.special import erfc, erfcinv

def ber_qam(snr, I, J):
    """
    Computes BER for rectangular QAM (I x J constellation).
    
    Parameters:
    - snr: Signal-to-Noise Ratio (linear scale).
    - I: Number of levels in I dimension.
    - J: Number of levels in J dimension.
    
    Returns:
    - BER: Bit-Error-Rate for the given SNR and QAM constellation.
    """
    log2_IJ = np.log2(I * J)
    coeff = np.sqrt((3) / (I**2 + J**2 - 2))
    term1 = ((I - 1) / I) * erfc(coeff * np.sqrt(snr))
    term2 = ((J - 1) / J) * erfc(coeff * np.sqrt(snr))
    return (1 / log2_IJ) * (term1 + term2)

def ber_psk(snr, lam):
    """
    Computes BER for PSK modulation.
    
    Parameters:
    - snr: Signal-to-Noise Ratio (linear scale).
    - lam: Number of bits per symbol.
    
    Returns:
    - BER: Bit-Error-Rate for the given SNR and PSK modulation.
    """
    return 0.5 * erfc(np.sqrt(snr / lam))

def snr_qam(ber, I, J):
    """
    Computes SNR for rectangular QAM (I x J constellation) given a desired BER.
    
    Parameters:
    - ber: Desired Bit-Error-Rate.
    - I: Number of levels in I dimension.
    - J: Number of levels in J dimension.
    
    Returns:
    - SNR: Signal-to-Noise Ratio (linear scale) for the given BER and QAM constellation.
    """
    log2_IJ = np.log2(I * J)
    a = np.sqrt((3) / (I**2 + J**2 - 2))
    C = (1 / log2_IJ) * ((I - 1) / I + (J - 1) / J)
    return ((erfcinv(ber / C) ** 2) / (a ** 2))

def snr_psk(ber, lam):
    """
    Computes SNR for PSK modulation given a desired BER.
    
    Parameters:
    - ber: Desired Bit-Error-Rate.
    - lam: Number of bits per symbol.
    
    Returns:
    - SNR: Signal-to-Noise Ratio (linear scale) for the given BER and PSK modulation.
    """
    return lam * (erfcinv(2 * ber))**2

def compute_snr_thresholds(modulation_formats, ber):
    """
    Compute SNR thresholds for various modulation formats given a target BER.
    
    Parameters:
    - modulation_formats: Dictionary of modulation formats and their parameters.
    - ber: Target Bit-Error-Rate.
    
    Returns:
    - snr_thresholds: Dictionary of SNR thresholds for each modulation format.
    """
    snr_thresholds = {}
    for mod, (lam, params) in modulation_formats.items():
        if isinstance(params, tuple):
            I, J = params
            snr_thresholds[mod] = snr_qam(ber, I, J)
        else:
            snr_thresholds[mod] = snr_psk(ber, lam)
    return snr_thresholds

def compute_gaussian_threshold(snr_thresholds):
    """
    Compute the Gaussian modulation threshold as the minimum of real modulations.
    
    Parameters:
    - snr_thresholds: Dictionary of SNR thresholds for each modulation format.
    
    Returns:
    - gaussian_threshold: Minimum SNR threshold among the real modulations.
    """
    min_snr = min(snr_thresholds.values())
    return 10 * np.log10(min_snr)