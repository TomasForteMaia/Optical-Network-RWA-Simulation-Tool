"""
config.py

This module defines the Config class, which holds all the essential configuration
parameters for the optical network simulation. It initializes fundamental constants,
system parameters, modulation formats, and bands while providing methods to load networks,
compute thresholds, and define bands.

Author: Tomás Maia 96340 MEEC
Date: 10/02/2025
"""

import numpy as np
from band_definition import define_bands
from modulation_ber_computation import compute_snr_thresholds, compute_gaussian_threshold
from network_analysis import load_networks_from_folders
import networkx as nx
import os

class Config:
    """
    Configuration class for the optical network simulation. 
    This class stores system parameters, initializes bands, computes SNR thresholds, 
    and loads network topologies.
    """
    def __init__(self):
        """
        Initialize configuration parameters including fundamental constants, 
        system parameters, modulation formats, and analysis settings.
        """

        # Fundamental constants
        self.h = 6.626e-34  # Planck's constant (J·s)
        self.c = 2.99792458e8  # Speed of light (m/s)

        # System parameters
        self.spacing = 64e9  # WDM channel spacing (Hz)
        self.Rs = self.spacing  # Symbol rate (Baud)
        self.Bch = self.spacing  # Channel bandwidth (Hz)
        self.Ls_max = 100  # Maximum Span Length (km) (Defines number of spans per link and also EDFA Gain)
        self.gamma = 1.27e-3  # Nonlinearity coefficient (1/W·m)
        self.D = 17 * 1e-12 / 1e-9 / 1e3  # Dispersion coefficient (s/m^2)
        self.S = 0.067 * 1e-12 / 1e-9 / 1e3 / 1e-9  # Dispersion slope (s/m^3)
        self.Cr = 0.028 / 1e3 / 1e12  # Raman gain slope in 1/W/m/Hz
        self.alpha_dB = 0.20  # Fiber attenuation (dB/km)

        # Simulation flags and setting
        self.verbose = True  # Enable detailed output
        self.add_drop = True  # Account for add/drop ROADMs
        self.use_gaussian_modulation = True  # Use Gaussian modulation assumption
        self.NMC = 1  # Number of Monte Carlo iterations for SNAP
        self.path_order_strategy = "shortest-first"  # Path order strategy: shortest-first, longest-first or random 
        self.wavelength_assignment_strategy = "inverse-first-fit"  # Wavelength assignment strategy: first-fit, inverse-first-fit, most-used, least-used or random
        self.k_paths = 3  # Number of paths to evaluate per demand
        self.ber = 1e-3  # Target BER (Bit-Error-Rate)
        self.OH = 0.28 # # Overhead factor (FEC and mapping overhead within BVTs)
        self.progressive_traffic = False  # Enable progressive traffic
        self.max_blocking_probability = 0.20  # Threshold for stopping progressive traffic

        # Modulation formats and bit-rates
        self.modulation_formats = {
            "BPSK": (1, 2), # Binary Phase-Shift Keying (lam=1, M=2)
            "QPSK": (2, 4), # Quadrature Phase-Shift Keying (lam=2, M=4)
            "8-QAM": (3, (4, 2)),  # Rectangular 8-ary Quadrature Amplitude Modulation: 4x2
            "16-QAM": (4, (4, 4)),  # Square 16-ary Quadrature Amplitude Modulation: 4x4
            "32-QAM": (5, (8, 4)),  # Rectangular 32-ary Quadrature Amplitude Modulation: 8x4
            "64-QAM": (6, (8, 8))  # Square 64-ary Quadrature Amplitude Modulation: 8x8
        }

        self.modulation_bit_rates = {
            mod: 2 * (self.Rs / (1 + self.OH)) * np.log2(m if isinstance(m, int) else np.prod(m)) / 1e9
            for mod, (_, m) in self.modulation_formats.items()
        }

        # Bands to analyze
        self.bands_to_analyze = ["Super C + Super L"]
        # self.bands_to_analyze = ["C", "Super C", "Super C + Super L"]
        # self.bands_to_analyze = ["C"]

        # Initialize attributes for bands, thresholds, and networks - To be initialized later
        self.all_bands = None
        self.snr_thresholds = None
        self.gaussian_threshold = None
        self.modulation_formats_with_thresholds = None
        self.networks = None

    def initialize_bands(self, contiguous=False):
        """
        Initialize the bands using the define_bands function.

        Parameters:
        - contiguous: Boolean flag indicating if the bands should be contiguous.
        """
        self.all_bands = define_bands(contiguous=contiguous, spacing=self.spacing, Bch=self.Bch, D=self.D)
        self.bands = {band: self.all_bands[band] for band in self.bands_to_analyze if band in self.all_bands}

    def initialize_thresholds(self):
        """
        Compute SNR thresholds for different modulation formats and 
        determine the Gaussian threshold for performance evaluation.
        """
        self.snr_thresholds = compute_snr_thresholds(self.modulation_formats, self.ber)
        self.gaussian_threshold = compute_gaussian_threshold(self.snr_thresholds)
        self.modulation_formats_with_thresholds = {mod: 10 * np.log10(snr) for mod, snr in self.snr_thresholds.items()}

    def load_networks(self, folders):
        """
        Load network topologies from specified folders containing .graphml files.
        
        Parameters:
        - folders (list of str): List of folder paths containing network files.
        """
        self.networks = load_networks_from_folders(folders)
        filter_network_names = ["CESNET.graphml", "COST239.graphml", "NSFNET.graphml", "DTGerman.graphml", "UBN.graphml", "PTbackbone.graphml", "CONUS30.graphml", "COST266.graphml", "CONUS60.graphml"]  # Networks to analyze
        # filter_network_names = ["UBN.graphml", "CONUS30.graphml", "CONUS60.graphml"]
        # filter_network_names = ["UBN.graphml", "CONUS60.graphml"]
        # filter_network_names = ["DTGerman.graphml"]
        # filter_network_names = ["CESNET.graphml", "CONUS60.graphml"]  # Networks to analyze
        # filter_network_names = ["NSFNET.graphml"]
        if filter_network_names:
            self.networks = {name: graph for name, graph in self.networks.items() if name in filter_network_names}
        if not self.networks:
            raise ValueError("No networks found. Please check the folder paths and network names.")

    def filter_networks_by_nodes(self, folder, node_counts, num_networks_per_count):
        """
        Load networks from the specified folder and filter them by node count.

        Parameters:
        - folder: Path to the folder containing .graphml files.
        - node_counts: List of node counts to filter the networks.
        - num_networks_per_count: Number of networks to select for each node count.

        Returns:
        - Dictionary of filtered networks.
        """
        self.networks = load_networks_from_folders([folder])

        # Sort networks first by the number of nodes, then by the number of edges
        sorted_networks = dict(sorted(self.networks.items(), key=lambda item: (len(item[1].nodes), len(item[1].edges))))

        filtered_networks = {}
        
        # Now filter the networks based on the specified node counts
        for node_count in node_counts:
            count = 0
            for name, G in sorted_networks.items():
                if len(G.nodes) == node_count:
                    filtered_networks[name] = G
                    count += 1
                    if count >= num_networks_per_count:
                        break
        
        if not filtered_networks:
            raise ValueError("No networks found with the specified node counts.")
        
        return filtered_networks

    def load_and_filter_networks_by_links(self, folders, num_networks_per_link, link_counts):
        """
        Load networks from the specified folders and filter them based on the number of links.

        Parameters:
        - folders: List of folder paths containing the graph files.
        - num_networks_per_link: Number of networks to select for each link count.
        - link_counts: List of link counts to filter the networks.

        Returns:
        - Dictionary with folder names as keys and lists of tuples (graph name, graph) as values.
        """
        all_graphs = {}
        for folder in folders:
            graphs = []
            count_per_link = {link_count: 0 for link_count in link_counts}
            for file_name in os.listdir(folder):
                if file_name.endswith('.graphml'):
                    graph = nx.read_graphml(os.path.join(folder, file_name))
                    num_links = len(graph.edges)
                    if num_links in link_counts and count_per_link[num_links] < num_networks_per_link:
                        graphs.append((file_name, graph))
                        count_per_link[num_links] += 1
                        if all(count >= num_networks_per_link for count in count_per_link.values()):
                            break
            all_graphs[folder] = graphs
        return all_graphs

    def filter_networks_by_nodes_and_links(self, folder, node_counts, num_links, num_networks_per_count):
        """
        Filter networks based on the number of nodes and links.

        Parameters:
        - folder: Path to the folder containing .graphml files.
        - node_counts: List of node counts to filter the networks.
        - num_links: Number of links to filter the networks.
        - num_networks_per_count: Number of networks to select for each node count.

        Returns:
        - Dictionary of filtered and sorted networks.
        """
        self.networks = load_networks_from_folders([folder])

        filtered_networks = {}
        
        for node_count in node_counts:
            count = 0
            for name, G in self.networks.items():
                if len(G.nodes) == node_count and len(G.edges) == num_links:
                    filtered_networks[name] = G
                    count += 1
                    if count >= num_networks_per_count:
                        break
        
        if not filtered_networks:
            raise ValueError("No networks found with the specified node counts and number of links.")
        
        # Sort the filtered networks first by the number of nodes and then by the number of links
        sorted_networks = dict(sorted(filtered_networks.items(), key=lambda item: (len(item[1].nodes), len(item[1].edges))))
        
        return sorted_networks

    def load_and_filter_networks_by_folder(self, folders, num_networks_per_folder):
        """
        Load networks from the specified folders and filter them based on the number of networks to consider in each folder.

        Parameters:
        - folders: List of folder paths containing the graph files.
        - num_networks_per_folder: Number of networks to select from each folder.

        Returns:
        - Dictionary with folder names as keys and lists of tuples (graph name, graph) as values.
        """
        all_graphs = {}
        for folder in folders:
            graphs = []
            count = 0
            for file_name in os.listdir(folder):
                if file_name.endswith('.graphml'):
                    graph = nx.read_graphml(os.path.join(folder, file_name))
                    graphs.append((file_name, graph))
                    count += 1
                    if count >= num_networks_per_folder:
                        break
            all_graphs[folder] = graphs
        return all_graphs

    # def load_and_filter_networks_by_folder(self, folders, num_networks_per_folder):
    #     """
    #     Load networks from the specified folders and filter them based on the number of networks to consider in each folder.

    #     Parameters:
    #     - folders: List of folder paths containing the graph files.
    #     - num_networks_per_folder: Number of networks to select from each folder.

    #     Returns:
    #     - Dictionary with graph names as keys and graphs as values.
    #     """
    #     all_graphs = {}
    #     for folder in folders:
    #         count = 0
    #         for file_name in os.listdir(folder):
    #             if file_name.endswith('.graphml'):
    #                 print(folder)
    #                 graph = nx.read_graphml(os.path.join(folder, file_name))
    #                 all_graphs[file_name] = graph  # Use graph name as key
    #                 count += 1
    #                 if count >= num_networks_per_folder:
    #                     break
        
    #     return all_graphs  # Return a flat dictionary of {graph_name: graph}

    def print_parameters(self):
        """
        Prints the simulation parameters in a human-readable format.
        """
        print("Simulation Parameters:")
        print(f"Fundamental Constants:")
        print(f"  Planck's constant (h): {self.h} J·s")
        print(f"  Speed of light (c): {self.c} m/s")
        
        print("\nSystem Parameters:")
        print(f"  WDM Channel Spacing (Spacing): {self.spacing} Hz")
        print(f"  Symbol Rate (Rs): {self.Rs} Baud")
        print(f"  Channel Bandwidth (Bch): {self.Bch} Hz")
        print(f"  Maximum Span Length (Ls_max): {self.Ls_max} km")
        print(f"  Nonlinearity Coefficient (Gamma): {self.gamma} 1/W·m")
        print(f"  Dispersion Coefficient (D): {self.D} s/m^2")
        print(f"  Dispersion Slope (S): {self.S} s/m^3")
        print(f"  Raman Gain Slope (Cr): {self.Cr} 1/W/m/Hz")
        print(f"  Fiber Attenuation (alpha_dB): {self.alpha_dB} dB/km")
        
        print("\nSimulation Flags and Settings:")
        print(f"  Verbose: {self.verbose}")
        print(f"  Add/Drop ROADMs: {self.add_drop}")
        print(f"  Use Gaussian Modulation: {self.use_gaussian_modulation}")
        print(f"  Number of Monte Carlo Iterations (NMC): {self.NMC}")
        print(f"  Path Order Strategy: {self.path_order_strategy}")
        print(f"  Wavelength Assignment Strategy: {self.wavelength_assignment_strategy}")
        print(f"  Number of Paths to Evaluate per Demand (k_paths): {self.k_paths}")
        print(f"  Target BER: {self.ber}")
        print(f"  Overhead Factor (OH): {self.OH}")
        print(f"  Progressive Traffic: {self.progressive_traffic}")
        print(f"  Max Blocking Probability: {self.max_blocking_probability}")
        
        print("\nModulation Formats:")
        for mod, (lam, m) in self.modulation_formats.items():
            print(f"  {mod}: λ = {lam}, M = {m}")
        
        print(f"\nBands to Analyze: {self.bands_to_analyze}")
        print(f"\nModulation Bit Rates: {self.modulation_bit_rates}")
        print(f"\nBand Details: {self.all_bands}")