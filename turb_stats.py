#!/usr/bin/env python3
# turb_stats.py
# This script processes computes first order turbulence statistics from time and space averaged data for channel flow simulations.

# import libraries ------------------------------------------------------------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import os
from tqdm import tqdm

# import modules --------------------------------------------------------------------------------------------------------------------------------------
import operations as op
import utils as ut

# =====================================================================================================================================================
# CONFIGURATION CLASSES
# =====================================================================================================================================================

@dataclass
class Config:
    """Configuration wrapper for all settings"""
    # File paths and cases
    folder_path: str
    input_format: str
    cases: List[str]
    timesteps: List[str]
    thermo_on: bool
    mhd_on: bool
    forcing: str
    Re: List[str]
    ref_temp: str

    # Output options
    ux_velocity_on: bool
    temp_on: bool
    tke_on: bool
    u_prime_sq_on: bool
    u_prime_v_prime_on: bool
    w_prime_sq_on: bool
    v_prime_sq_on: bool

    # TKE Budget options
    tke_budget_on: bool
    tke_production_on: bool
    tke_dissipation_on: bool
    tke_convection_on: bool
    tke_viscous_diffusion_on: bool
    tke_pressure_transport_on: bool
    tke_turbulent_diffusion_on: bool

    # Processing options
    norm_by_u_tau_sq: bool
    norm_ux_by_u_tau: bool
    norm_y_to_y_plus: bool
    norm_temp_by_ref_temp: bool

    # Plotting options
    half_channel_plot: bool
    linear_y_scale: bool
    log_y_scale: bool
    multi_plot: bool
    display_fig: bool
    save_fig: bool
    save_to_path: bool
    plot_name: str

    # Reference data options
    ux_velocity_log_ref_on: bool
    mhd_NK_ref_on: bool
    mkm180_ch_ref_on: bool

    @classmethod
    def from_module(cls, config_module):
        """Create Config from imported config module"""
        return cls(
            folder_path=getattr(config_module, 'folder_path', ''),
            input_format=getattr(config_module, 'input_format', 'dat'),
            cases=getattr(config_module, 'cases', []),
            timesteps=getattr(config_module, 'timesteps', []),
            thermo_on=getattr(config_module, 'thermo_on', False),
            mhd_on=getattr(config_module, 'mhd_on', False),
            forcing=getattr(config_module, 'forcing', 'CMF'),
            Re=getattr(config_module, 'Re', [1.0]),
            ref_temp=getattr(config_module, 'ref_temp', [300.0]),
            ux_velocity_on=getattr(config_module, 'ux_velocity_on', False),
            temp_on=getattr(config_module, 'temp_on', False),
            tke_on=getattr(config_module, 'tke_on', False),
            u_prime_sq_on=getattr(config_module, 'u_prime_sq_on', False),
            u_prime_v_prime_on=getattr(config_module, 'u_prime_v_prime_on', False),
            w_prime_sq_on=getattr(config_module, 'w_prime_sq_on', False),
            v_prime_sq_on=getattr(config_module, 'v_prime_sq_on', False),
            tke_budget_on=getattr(config_module, 'tke_budget_on', False),
            tke_production_on=getattr(config_module, 'tke_production_on', False),
            tke_dissipation_on=getattr(config_module, 'tke_dissipation_on', False),
            tke_convection_on=getattr(config_module, 'tke_convection_on', False),
            tke_viscous_diffusion_on=getattr(config_module, 'tke_viscous_diffusion_on', False),
            tke_pressure_transport_on=getattr(config_module, 'tke_pressure_transport_on', False),
            tke_turbulent_diffusion_on=getattr(config_module, 'tke_turbulent_diffusion_on', False),
            norm_by_u_tau_sq=getattr(config_module, 'norm_by_u_tau_sq', False),
            norm_ux_by_u_tau=getattr(config_module, 'norm_ux_by_u_tau', False),
            norm_y_to_y_plus=getattr(config_module, 'norm_y_to_y_plus', False),
            norm_temp_by_ref_temp=getattr(config_module, 'norm_temp_by_ref_temp', False),
            half_channel_plot=getattr(config_module, 'half_channel_plot', False),
            linear_y_scale=getattr(config_module, 'linear_y_scale', True),
            log_y_scale=getattr(config_module, 'log_y_scale', False),
            multi_plot=getattr(config_module, 'multi_plot', True),
            display_fig=getattr(config_module, 'display_fig', False),
            save_fig=getattr(config_module, 'save_fig', True),
            save_to_path=getattr(config_module, 'save_to_path', False),
            plot_name=getattr(config_module, 'plot_name', ''),
            ux_velocity_log_ref_on=getattr(config_module, 'ux_velocity_log_ref_on', False),
            mhd_NK_ref_on=getattr(config_module, 'mhd_NK_ref_on', False),
            mkm180_ch_ref_on=getattr(config_module, 'mkm180_ch_ref_on', False),
        )

@dataclass
class PlotConfig:
    """Configuration for plotting aesthetics"""

    colors_1: Dict[str, str] = None
    colors_2: Dict[str, str] = None
    colors_3: Dict[str, str] = None
    colors_4: Dict[str, str] = None
    colors_blck: Dict[str, str] = None
    stat_labels: Dict[str, str] = None

    def __post_init__(self):
        if self.colors_1 is None:
            self.colors_1 = {
                'ux_velocity': 'b',
                'u_prime_sq': 'r',
                'u_prime_v_prime': 'g',
                'w_prime_sq': 'c',
                'v_prime_sq': 'm',
            }

        if self.colors_2 is None:
            self.colors_2 = {
                'ux_velocity': 'r',
                'u_prime_sq': 'orange',
                'u_prime_v_prime': 'lime',
                'w_prime_sq': 'b',
                'v_prime_sq': 'purple',
            }

        if self.colors_3 is None:
            self.colors_3 = {
                'ux_velocity': 'indigo',
                'u_prime_sq': 'maroon',
                'u_prime_v_prime': 'olive',
                'w_prime_sq': 'navy',
                'v_prime_sq': 'deeppink',
            }

        if self.colors_4 is None:
            self.colors_4 = {
                'ux_velocity': 'magenta',
                'u_prime_sq': 'peru',
                'u_prime_v_prime': 'lightgreen',
                'w_prime_sq': 'teal',
                'v_prime_sq': 'indigo',
            }

        if self.colors_blck is None:
            self.colors_blck = {
                'ux_velocity': 'black',
                'u_prime_sq': 'black',
                'u_prime_v_prime': 'black',
                'w_prime_sq': 'black',
                'v_prime_sq': 'black',
            }

        if self.stat_labels is None:
            self.stat_labels = {
                "ux_velocity": "Streamwise Velocity",
                "u_prime_sq": "<u'u'>",
                "u_prime_v_prime": "<u'v'>",
                "v_prime_sq": "<v'v'>",
                "w_prime_sq": "<w'w'>",
                # TKE Budget terms
                "production": "Production",
                "dissipation": "Dissipation",
                "convection": "Convection",
                "viscous_diffusion": "Viscous Diffusion",
                "pressure_transport": "Pressure Transport",
                "turbulent_diffusion": "Turbulent Diffusion",
                "buoyancy": "Buoyancy",
                "mhd": "MHD (Lorentz)",
            }

    @property
    def colours(self):
        """Returns tuple of all color schemes"""
        return (self.colors_1, self.colors_2, self.colors_3, self.colors_4, self.colors_blck)
    def colours_ref(self):
        """Returns tuple of all color schemes with black first for plotting a reference"""
        return (self.colors_blck, self.colors_1, self.colors_2, self.colors_3, self.colors_4)

# =====================================================================================================================================================
# DATA LOADING & MANAGEMENT
# =====================================================================================================================================================

def create_data_loader(config: Config, data_types: List[str] = None):
    """
    Factory function to create the appropriate data loader based on input_format.

    Args:
        config: Configuration object
        data_types: List of XDMF data types to load. Only used for XDMF format.
                    Valid types: 'inst', 't_avg', 'tsp_avg', or specific combinations
                    like 'tsp_avg_flow', 't_avg_thermo', etc.
                    Defaults to ['tsp_avg'] for turb_stats (time-space averaged data).

    Returns:
        Data loader instance (TurbulenceTextData or TurbulenceXDMFData)
    """
    fmt = config.input_format.lower()

    if fmt in ['xdmf', 'visu']:
        # Determine which data types to load based on enabled features
        tke_budget_enabled = (config.tke_budget_on or config.tke_production_on or 
                              config.tke_dissipation_on or config.tke_convection_on or 
                              config.tke_viscous_diffusion_on or config.tke_pressure_transport_on or 
                              config.tke_turbulent_diffusion_on)
        
        if data_types is None:
            if tke_budget_enabled:
                # For TKE budget, we need t_avg (for gradients) and tsp_avg (for dissipation dudu terms)
                # Load both to ensure all data is available
                data_types = ['t_avg', 'tsp_avg']
            else:
                # For regular statistics only, use tsp_avg
                data_types = ['tsp_avg']
        elif tke_budget_enabled:
            # Ensure both t_avg and tsp_avg are loaded when TKE budget is enabled
            data_types = list(data_types)
            if 't_avg' not in data_types:
                data_types.append('t_avg')
            if 'tsp_avg' not in data_types:
                data_types.append('tsp_avg')
        print(f"Using XDMF data loader with data_types={data_types}...")
        return TurbulenceXDMFData(
            config.folder_path,
            config.cases,
            config.timesteps,
            data_types=data_types
        )
    elif fmt in ['dat', 'text']:
        print("Using text (.dat) data loader...")
        return TurbulenceTextData(
            config.folder_path,
            config.cases,
            config.timesteps,
            config.thermo_on
        )
    else:
        raise ValueError(f"Unknown input_format: '{config.input_format}'. Must be 'dat', 'text', 'xdmf', or 'visu'")
    

class TurbulenceTextData:
    """Manages time-space averaged data loading and access from .dat files"""

    def __init__(self, folder_path: str, cases: List[str], timesteps: List[str], thermo_on: bool):
        self.folder_path = folder_path
        self.cases = cases
        self.timesteps = timesteps
        self.quantities = ut.get_quantities(thermo_on)
        # Nested structure: {case_timestep: {quantity: array}}
        self.data: Dict[str, Dict[str, np.ndarray]] = {}

    def load_all(self) -> None:
        """Load all time-space averaged data files"""
        for case in self.cases:
            for timestep in self.timesteps:
                # Create nested key for this case/timestep
                key = f"{case}_{timestep}"
                if key not in self.data:
                    self.data[key] = {}

                for quantity in self.quantities:
                    self._load_single(case, quantity, timestep)

    def _load_single(self, case: str, quantity: str, timestep: str) -> None:
        """Load a single data file"""
        key = f"{case}_{timestep}"
        file_path = ut.data_filepath(self.folder_path, case, quantity, timestep)

        print(f"Looking for files in: {file_path}")

        if os.path.isfile(file_path):
            data = ut.load_ts_avg_data(file_path)
            if data is not None:
                if key not in self.data:
                    self.data[key] = {}
                self.data[key][quantity] = data
            else:
                print(f'.dat file is empty for {case}, {timestep}, {quantity}')
        else:
            print(f'No .dat file found for {case}, {timestep}, {quantity}')

    def get(self, case: str, quantity: str, timestep: str) -> Optional[np.ndarray]:
        """Get specific data array"""
        key = f"{case}_{timestep}"
        if key in self.data and quantity in self.data[key]:
            return self.data[key][quantity]
        return None

    def has(self, case: str, quantity: str, timestep: str) -> bool:
        """Check if data exists"""
        key = f"{case}_{timestep}"
        return key in self.data and quantity in self.data[key]

    def keys(self):
        """Return all data keys (case_timestep format)"""
        return self.data.keys()

class TurbulenceXDMFData:
    """Manages all data from .xdmf files"""

    def __init__(self, folder_path: str, cases: List[str], timesteps: List[str], data_types: List[str] = None):
        self.folder_path = folder_path
        self.cases = cases
        self.timesteps = timesteps
        self.data_types = data_types
        # Nested structure: {case_timestep: {variable: array}}
        self.data: Dict[str, Dict[str, np.ndarray]] = {}
        self.grid_info: Dict = {}

    def load_all(self) -> None:
        """Load all XDMF files for all cases and timesteps"""
        for case in self.cases:
            for timestep in self.timesteps:
                self._load_single(case, timestep)

    def _load_single(self, case: str, timestep: str) -> None:
        """Load XDMF files for a single case and timestep"""
        key = f"{case}_{timestep}"

        # Get file paths for this case/timestep
        file_names = ut.visu_file_paths(self.folder_path, case, timestep)

        # Check which files exist
        existing_files = [f for f in file_names if os.path.isfile(f)]

        if not existing_files:
            print(f'No .xdmf files found for {case}, {timestep}')
            return

        # Load the XDMF files (filtered by data_types if specified)
        arrays, grid_info = ut.xdmf_reader_wrapper(file_names, case=case, timestep=timestep, data_types=self.data_types)

        if arrays and key in arrays:
            # Store grid info if not already stored
            if not self.grid_info and grid_info:
                self.grid_info = grid_info

            # Convert 3D XDMF arrays to 2D format expected by statistics code
            # Format: [index, y_coord, value] matching text file format
            raw_data = arrays[key]
            converted_data = {}

            # Get y-coordinates from grid info (node coordinates)
            y_nodes = grid_info.get('grid_y', None) if grid_info else None

            for var_name, var_array in raw_data.items():
                if var_array.ndim == 3:
                    # 3D array (z, y, x) - extract y-profile from averaged data
                    # For tsp_avg data, values are constant along x and z, so take [0, :, 0]
                    y_profile = var_array[0, :, 0]
                    n_points = len(y_profile)

                    # Create 2D array with columns [index, y_coord, value]
                    converted = np.zeros((n_points, 3))
                    converted[:, 0] = np.arange(n_points)  # Index

                    # Handle y-coordinates: may need cell centers if data is cell-centered
                    if y_nodes is not None:
                        if len(y_nodes) == n_points:
                            # Node-centered data
                            converted[:, 1] = y_nodes
                        elif len(y_nodes) == n_points + 1:
                            # Cell-centered data - compute cell centers
                            y_centers = 0.5 * (y_nodes[:-1] + y_nodes[1:])
                            converted[:, 1] = y_centers
                        else:
                            # Fallback to linear spacing
                            converted[:, 1] = np.linspace(-1, 1, n_points)
                    else:
                        converted[:, 1] = np.linspace(-1, 1, n_points)  # Default normalized y

                    converted[:, 2] = y_profile  # Values
                    converted_data[var_name] = converted

                elif var_array.ndim == 1:
                    # Already 1D - create 2D format
                    n_points = len(var_array)
                    converted = np.zeros((n_points, 3))
                    converted[:, 0] = np.arange(n_points)
                    if y_nodes is not None:
                        if len(y_nodes) == n_points:
                            converted[:, 1] = y_nodes
                        elif len(y_nodes) == n_points + 1:
                            converted[:, 1] = 0.5 * (y_nodes[:-1] + y_nodes[1:])
                        else:
                            converted[:, 1] = np.linspace(-1, 1, n_points)
                    else:
                        converted[:, 1] = np.linspace(-1, 1, n_points)
                    converted[:, 2] = var_array
                    converted_data[var_name] = converted
                else:
                    # Keep as-is for other dimensions
                    converted_data[var_name] = var_array

            self.data[key] = converted_data
            print(f"Loaded {len(converted_data)} variables from XDMF files for {case}, {timestep}")
        else:
            print(f'No arrays extracted from XDMF files for {case}, {timestep}')

    def get(self, case: str, variable: str, timestep: str) -> Optional[np.ndarray]:
        """Get specific data array"""
        key = f"{case}_{timestep}"
        if key in self.data and variable in self.data[key]:
            return self.data[key][variable]
        return None

    def has(self, case: str, variable: str, timestep: str) -> bool:
        """Check if data exists"""
        key = f"{case}_{timestep}"
        return key in self.data and variable in self.data[key]

    def keys(self):
        """Return all data keys (case_timestep format)"""
        return self.data.keys()

    def get_variables(self, case: str, timestep: str) -> List[str]:
        """Get list of all variables available for a case/timestep"""
        key = f"{case}_{timestep}"
        if key in self.data:
            return list(self.data[key].keys())
        return []

class ReferenceData:
    """Manages all reference datasets"""

    # Reference data file paths
    REF_MKM180_MEANS_PATH = 'Reference_Data/MKM180_profiles/chan180.means'
    REF_MKM180_REYSTRESS_PATH = 'Reference_Data/MKM180_profiles/chan180.reystress'

    NK_REF_PATHS = {
        'ref_NK_Ha_6': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_6_turb.txt',
        'ref_NK_Ha_4': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_4_turb.txt',
        'ref_NK_uu12_Ha_6': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_6_uu12_rms.txt',
        'ref_NK_uu12_Ha_4': 'Reference_Data/Noguchi&Kasagi_mhd_ref_data/thtlabs_Ha_4_uu12_rms.txt',
    }

    #REF_XCOMP_HA_6_PATH = 'Reference_Data/XCompact3D_mhd_validation/u_prime_sq.txt'

    def __init__(self, config: Config):
        self.config = config
        self.mkm180_stats: Optional[Dict[str, np.ndarray]] = None
        self.mkm180_y: Optional[np.ndarray] = None
        self.mkm180_y_plus: Optional[np.ndarray] = None

        self.NK_H4_stats: Optional[Dict[str, np.ndarray]] = None
        self.NK_H6_stats: Optional[Dict[str, np.ndarray]] = None
        self.NK_ref_y_H4: Optional[np.ndarray] = None
        self.NK_ref_y_H6: Optional[np.ndarray] = None
        self.NK_ref_y_uu12_H4: Optional[np.ndarray] = None
        self.NK_ref_y_uu12_H6: Optional[np.ndarray] = None

        self.xcomp_H6_stats: Optional[Dict[str, np.ndarray]] = None
        self.xcomp_yplus_uu11_H6: Optional[np.ndarray] = None

    def load_all(self) -> None:
        """Load all enabled reference datasets"""
        if self.config.mkm180_ch_ref_on:
            self._load_mkm180()

        if self.config.mhd_NK_ref_on:
            self._load_noguchi_kasagi()

        #if self.config.mhd_XCompact_ref_on:
        #    self._load_xcompact()

    def _load_mkm180(self) -> None:
        """Load MKM180 reference data"""
        try:
            ref_means = np.loadtxt(self.REF_MKM180_MEANS_PATH)
            ref_reystress = np.loadtxt(self.REF_MKM180_REYSTRESS_PATH)

            self.mkm180_y = ref_means[:, 0]
            self.mkm180_y_plus = ref_means[:, 1]

            self.mkm180_stats = {
                'ux_velocity': ref_means[:, 2],
                'u_prime_sq': ref_reystress[:, 2],
                'v_prime_sq': ref_reystress[:, 3],
                'w_prime_sq': ref_reystress[:, 4],
                'u_prime_v_prime': ref_reystress[:, 5]
            }
            print("mkm180 reference data loaded successfully.")
        except Exception as e:
            print(f"mkm180 reference is disabled or required data is missing: {e}")

    def _load_noguchi_kasagi(self) -> None:
        """Load Noguchi & Kasagi MHD reference data"""
        try:
            NK_data = {}
            for key, path in self.NK_REF_PATHS.items():
                NK_data[key] = np.loadtxt(path)

            self.NK_ref_y_H4 = NK_data['ref_NK_Ha_4'][:, 1]
            self.NK_ref_y_H6 = NK_data['ref_NK_Ha_6'][:, 1]
            self.NK_ref_y_uu12_H4 = NK_data['ref_NK_uu12_Ha_4'][:, 1]
            self.NK_ref_y_uu12_H6 = NK_data['ref_NK_uu12_Ha_6'][:, 1]

            self.NK_H4_stats = {
                'ux_velocity': NK_data['ref_NK_Ha_4'][:, 2] * 1.02169,
                'u_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 3]),
                'u_prime_v_prime': -1 * NK_data['ref_NK_uu12_Ha_4'][:, 2],
                'v_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 4]),
                'w_prime_sq': np.square(NK_data['ref_NK_Ha_4'][:, 5])
            }

            self.NK_H6_stats = {
                'ux_velocity': NK_data['ref_NK_Ha_6'][:, 2],
                'u_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 3]),
                'u_prime_v_prime': -1 * NK_data['ref_NK_uu12_Ha_6'][:, 2],
                'v_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 4]),
                'w_prime_sq': np.square(NK_data['ref_NK_Ha_6'][:, 5])
            }
            print("Noguchi & Kasagi MHD channel reference data loaded successfully.")
        except Exception as e:
            print(f"NK mhd reference is disabled or required data is missing: {e}")

# =====================================================================================================================================================
# REYNOLDS STRESS CLASSES
# =====================================================================================================================================================

class ReStresses(ABC):
    """Abstract base class for Reynolds stress statistics"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the statistic from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceTextData) -> bool:
        """Compute statistic for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute statistic
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]



class ReynoldsStressuu11(ReStresses):
    """Reynolds stress u'u'"""

    def __init__(self):
        super().__init__('u_prime_sq', "<u'u'>", ['u1', 'uu11'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_normal_stress(data_dict['u1'], data_dict['uu11'])

class ReynoldsStressuu12(ReStresses):
    """Reynolds stress u'v'"""

    def __init__(self):
        super().__init__('u_prime_v_prime', "<u'v'>", ['u1', 'u2', 'uu12'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_shear_stress(data_dict['u1'], data_dict['u2'], data_dict['uu12'])

class ReynoldsStressuu22(ReStresses):
    """Reynolds stress v'v'"""

    def __init__(self):
        super().__init__('v_prime_sq', "<v'v'>", ['u2', 'uu22'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_normal_stress(data_dict['u2'], data_dict['uu22'])

class ReynoldsStressuu33(ReStresses):
    """Reynolds stress w'w'"""

    def __init__(self):
        super().__init__('w_prime_sq', "<w'w'>", ['u3', 'uu33'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.compute_normal_stress(data_dict['u3'], data_dict['uu33'])

# =====================================================================================================================================================
# PROFILE CLASSES
# =====================================================================================================================================================

class Profiles(ABC):
    """Abstract base class for velocity and temperature profiles"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the profile from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceTextData) -> bool:
        """Compute profile for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute profile
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]


class StreamwiseVelocity(Profiles):
    """Streamwise velocity profile (u1)"""

    def __init__(self):
        super().__init__('ux_velocity', 'Streamwise Velocity', ['u1'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return op.read_profile(data_dict['u1'])


class Temperature(Profiles):
    """Temperature profile"""

    def __init__(self, norm_temp_by_ref_temp: bool, ref_temp: float):
        super().__init__('temperature', 'Temperature', ['T'])
        self.norm_temp_by_ref_temp = norm_temp_by_ref_temp
        self.ref_temp = ref_temp

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        if self.norm_temp_by_ref_temp:
            return op.read_profile(data_dict['T'])
        else:
            undim_temp = op.read_profile(data_dict['T'])
            return undim_temp * self.ref_temp


class TurbulentKineticEnergy(Profiles):
    """Turbulent Kinetic Energy (TKE)"""

    def __init__(self):
        super().__init__('TKE', 'k', ['u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'])

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        u_prime_sq = op.compute_normal_stress(data_dict['u1'], data_dict['uu11'])
        v_prime_sq = op.compute_normal_stress(data_dict['u2'], data_dict['uu22'])
        w_prime_sq = op.compute_normal_stress(data_dict['u3'], data_dict['uu33'])
        return op.compute_tke(u_prime_sq, v_prime_sq, w_prime_sq)

# =====================================================================================================================================================
# TKE BUDGET CLASSES
# =====================================================================================================================================================

class TkeBudget(ABC):
    """Abstract base class for TKE budget terms"""

    def __init__(self, name: str, label: str, required_quantities: List[str]):
        self.name = name
        self.label = label
        self.required_quantities = required_quantities
        self.raw_results: Dict[Tuple[str, str], np.ndarray] = {}
        self.processed_results: Dict[Tuple[str, str], np.ndarray] = {}

    @abstractmethod
    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute the TKE budget term from required data"""
        pass

    def compute_for_case(self, case: str, timestep: str, data_loader: TurbulenceTextData) -> bool:
        """Compute TKE budget term for a specific case and timestep"""
        # Gather required data
        data_dict = {}
        for quantity in self.required_quantities:
            if not data_loader.has(case, quantity, timestep):
                print(f"Missing {quantity} data for {self.name} calculation: {case}, {timestep}")
                return False
            data_dict[quantity] = data_loader.get(case, quantity, timestep)

        # Compute budget term
        result = self.compute(data_dict)
        self.raw_results[(case, timestep)] = result
        return True

    def get_half_domain(self, values: np.ndarray) -> np.ndarray:
        """Get first half of domain"""
        return values[:(len(values)//2)]


class TKE_Production(TkeBudget):
    """
    TKE Production term: P = -⟨u'ᵢu'ⱼ⟩ ∂Uᵢ/∂xⱼ

    For fully developed channel flow:
    - Homogeneity: ∂Uᵢ/∂x = ∂Uᵢ/∂z = 0
    - Mean velocities: U₁ = U(y), U₂ = 0, U₃ = 0 → only ∂U₁/∂y is non-zero
    - Dominant term: P ≈ -⟨u'v'⟩ ∂U/∂y
    - Full expression with all terms computed for numerical accuracy
    """

    def __init__(self):
        super().__init__(
            'production',
            'Production',
            ['u1', 'u2', 'u3', 'uu11', 'uu12', 'uu13', 'uu22', 'uu23', 'uu33']
        )

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Compute velocity gradients from t_avg data
        grad_dict = op.compute_velocity_gradients(data_dict)
        data_dict.update(grad_dict)

        # Build Reynolds stress dictionary for existing compute_production function
        tke_comp_dict = {
            'u1p_u1p': data_dict['uu11'][:, 2] - np.square(data_dict['u1'][:, 2]),
            'u1p_u2p': data_dict['uu12'][:, 2] - data_dict['u1'][:, 2] * data_dict['u2'][:, 2],
            'u1p_u3p': data_dict['uu13'][:, 2] - data_dict['u1'][:, 2] * data_dict['u3'][:, 2],
            'u2p_u1p': data_dict['uu12'][:, 2] - data_dict['u1'][:, 2] * data_dict['u2'][:, 2],
            'u2p_u2p': data_dict['uu22'][:, 2] - np.square(data_dict['u2'][:, 2]),
            'u2p_u3p': data_dict['uu23'][:, 2] - data_dict['u2'][:, 2] * data_dict['u3'][:, 2],
            'u3p_u1p': data_dict['uu13'][:, 2] - data_dict['u1'][:, 2] * data_dict['u3'][:, 2],
            'u3p_u2p': data_dict['uu23'][:, 2] - data_dict['u2'][:, 2] * data_dict['u3'][:, 2],
            'u3p_u3p': data_dict['uu33'][:, 2] - np.square(data_dict['u3'][:, 2]),
            # Channel flow: homogeneous in x and z, so ∂/∂x = ∂/∂z = 0
            'du1dx': np.zeros_like(data_dict['u1'][:, 2]),
            'du1dy': data_dict['du1dy'][:, 2],
            'du1dz': np.zeros_like(data_dict['u1'][:, 2]),
            'du2dx': np.zeros_like(data_dict['u1'][:, 2]),
            'du2dy': data_dict['du2dy'][:, 2],
            'du2dz': np.zeros_like(data_dict['u1'][:, 2]),
            'du3dx': np.zeros_like(data_dict['u1'][:, 2]),
            'du3dy': data_dict['du3dy'][:, 2],
            'du3dz': np.zeros_like(data_dict['u1'][:, 2]),
        }
        result = op.compute_production(tke_comp_dict)
        return result['prod_total']


class TKE_Dissipation(TkeBudget):
    """
    TKE Dissipation term: ε = -(1/Re) ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩

    Represents the conversion of TKE to internal energy through viscous action.
    """

    def __init__(self, Re: float):
        super().__init__(
            'dissipation',
            'Dissipation',
            ['dudu11', 'dudu12', 'dudu13', 'dudu22', 'dudu23', 'dudu33']
        )
        self.Re = Re

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Build dictionary with dissipation correlations
        tke_comp_dict = {
            'dudu11': data_dict['dudu11'][:, 2],
            'dudu12': data_dict['dudu12'][:, 2],
            'dudu13': data_dict['dudu13'][:, 2],
            'dudu22': data_dict['dudu22'][:, 2],
            'dudu23': data_dict['dudu23'][:, 2],
            'dudu33': data_dict['dudu33'][:, 2],
        }

        result = op.compute_dissipation(self.Re, tke_comp_dict)
        return result['dissipation']


class TKE_Convection(TkeBudget):
    """
    TKE Convection (Advection) term: C = -Uⱼ ∂k/∂xⱼ

    Represents the transport of TKE by the mean flow.
    
    For fully developed channel flow:
    - Homogeneity: ∂k/∂x = ∂k/∂z = 0
    - Mean velocities: U₁ = U(y), U₂ = 0, U₃ = 0
    - Therefore: C = -U₁·0 - U₂·∂k/∂y - U₃·0 = 0 (since U₂ = 0)
    
    Note: We compute -U₂∂k/∂y anyway to account for small numerical deviations from U₂ = 0
    """

    def __init__(self):
        super().__init__(
            'convection',
            'Convection',
            ['u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33']
        )

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Compute TKE gradients from t_avg data
        grad_dict = op.compute_tke_gradients(data_dict)
        data_dict.update(grad_dict)

        # Build gradient dictionary for existing compute_convection function
        mean_velocities = {
            'U1': data_dict['u1'][:, 2],
            'U2': data_dict['u2'][:, 2],
            'U3': data_dict['u3'][:, 2],
        }
        tke_gradients = {
            'dkdx': np.zeros_like(data_dict['u1'][:, 2]),  # Channel: homogeneous in x
            'dkdy': data_dict['dkdy'][:, 2],                # Only y-gradient is non-zero
            'dkdz': np.zeros_like(data_dict['u1'][:, 2]),  # Channel: homogeneous in z
        }
        result = op.compute_convection(mean_velocities, tke_gradients)
        return result['convection']


class TKE_ViscousDiffusion(TkeBudget):
    """
    TKE Viscous Diffusion term: VD = (1/Re) ∂²k/∂xⱼ²

    Represents the diffusion of TKE by molecular viscosity.
    """

    def __init__(self, Re: float):
        super().__init__(
            'viscous_diffusion',
            'Viscous Diffusion',
            ['u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33']
        )
        self.Re = Re

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Compute TKE gradients from t_avg data
        grad_dict = op.compute_tke_gradients(data_dict)
        data_dict.update(grad_dict)

        # Build second derivative dictionary for existing compute_viscous_diffusion function
        tke_second_derivs = {
            'd2kdx2': np.zeros_like(data_dict['u1'][:, 2]),  # Channel: homogeneous in x
            'd2kdy2': data_dict['d2kdy2'][:, 2],              # Only y-derivative is non-zero
            'd2kdz2': np.zeros_like(data_dict['u1'][:, 2]),  # Channel: homogeneous in z
        }
        result = op.compute_viscous_diffusion(self.Re, tke_second_derivs)
        return result['viscous_diffusion']


class TKE_PressureTransport(TkeBudget):
    """
    TKE Pressure Transport term: PT = -∂⟨p'u'ⱼ⟩/∂xⱼ

    Represents the redistribution of TKE by pressure fluctuations.
    Also called pressure diffusion or velocity-pressure gradient correlation.
    For channel flow: PT = -∂⟨p'v'⟩/∂y (computed from t_avg)
    """

    def __init__(self):
        super().__init__(
            'pressure_transport',
            'Pressure Transport',
            ['pr', 'u2', 'pru2']
        )

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Compute pressure gradient from t_avg data
        grad_dict = op.compute_pressure_gradient(data_dict)
        if not grad_dict:
            return np.zeros_like(data_dict['u2'][:, 2])
        data_dict.update(grad_dict)

        # Build gradient dictionary for existing compute_pressure_transport function
        pressure_velocity_corr_grads = {
            'd_pu1p_dx': np.zeros_like(data_dict['u2'][:, 2]),  # Channel: homogeneous in x
            'd_pu2p_dy': data_dict['d_pu2p_dy'][:, 2],          # Only y-derivative is non-zero
            'd_pu3p_dz': np.zeros_like(data_dict['u2'][:, 2]),  # Channel: homogeneous in z
        }
        result = op.compute_pressure_transport(pressure_velocity_corr_grads)
        return result['pressure_transport']


class TKE_TurbulentDiffusion(TkeBudget):
    """
    TKE Turbulent Diffusion term: TD = -(1/2) ∂⟨u'ᵢu'ᵢu'ⱼ⟩/∂xⱼ

    Represents the transport of TKE by turbulent velocity fluctuations.
    Also called turbulent transport.
    """

    def __init__(self):
        super().__init__(
            'turbulent_diffusion',
            'Turbulent Diffusion',
            ['u1', 'u2', 'u3', 'uu11', 'uu12', 'uu22', 'uu23', 'uu33',
             'uuu112', 'uuu222', 'uuu233']
        )

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # Compute triple correlation gradient from t_avg data
        grad_dict = op.compute_triple_correlation_gradient(data_dict)
        if not grad_dict:
            return np.zeros_like(data_dict['u1'][:, 2])
        data_dict.update(grad_dict)

        # Build gradient dictionary for existing compute_turbulent_diffusion function
        triple_corr_grads = {
            'd_uiuiu1_dx': np.zeros_like(data_dict['u1'][:, 2]),  # Channel: homogeneous in x
            'd_uiuiu2_dy': data_dict['d_uiuiu2_dy'][:, 2],        # Only y-derivative is non-zero
            'd_uiuiu3_dz': np.zeros_like(data_dict['u1'][:, 2]),  # Channel: homogeneous in z
        }
        result = op.compute_turbulent_diffusion(triple_corr_grads)
        return result['turbulent_diffusion']


# Placeholder classes for future implementation
class TKE_Buoyancy(TkeBudget):
    """
    TKE Buoyancy term: B = -β gᵢ ⟨u'ᵢT'⟩

    Represents the production/destruction of TKE due to buoyancy effects.
    For vertical direction (gravity in y): B = -β g ⟨v'T'⟩

    TODO: Implement when thermal data is available.
    """

    def __init__(self):
        super().__init__(
            'buoyancy',
            'Buoyancy',
            ['u2', 'T', 'u2T']  # Placeholder - actual requirements TBD
        )

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("Buoyancy term not yet implemented")


class TKE_MHD(TkeBudget):
    """
    TKE MHD (Lorentz force) term: M = ⟨u'ᵢj'ᵢ⟩ × B / ρ

    Represents the interaction between velocity fluctuations and
    induced current fluctuations in an MHD flow.

    TODO: Implement when MHD data is available.
    """

    def __init__(self):
        super().__init__(
            'mhd',
            'MHD (Lorentz)',
            ['u1', 'u2', 'u3', 'j1', 'j2', 'j3']  # Placeholder - actual requirements TBD
        )

    def compute(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("MHD term not yet implemented")


# =====================================================================================================================================================
# PIPELINE CLASS
# =====================================================================================================================================================

class TurbulenceStatsPipeline:
    """Orchestrates the computation and processing of turbulence statistics"""

    def __init__(self, config: Config, data_loader: TurbulenceTextData):
        self.config = config
        self.data_loader = data_loader
        self.statistics: List[Union[ReStresses, Profiles, TkeBudget]] = []
        self._register_statistics()

    def _register_statistics(self) -> None:
        """Register all enabled statistics based on configuration"""
        if self.config.ux_velocity_on:
            self.statistics.append(StreamwiseVelocity())

        if self.config.u_prime_sq_on:
            self.statistics.append(ReynoldsStressuu11())

        if self.config.u_prime_v_prime_on:
            self.statistics.append(ReynoldsStressuu12())

        if self.config.v_prime_sq_on:
            self.statistics.append(ReynoldsStressuu22())

        if self.config.w_prime_sq_on:
            self.statistics.append(ReynoldsStressuu33())

        if self.config.tke_on:
            self.statistics.append(TurbulentKineticEnergy())

        if self.config.temp_on:
            self.statistics.append(Temperature(self.config.norm_temp_by_ref_temp, self.config.ref_temp))

        # TKE Budget terms
        # Get Reynolds number for terms that require it
        Re = float(self.config.Re[0]) if self.config.Re else 1.0

        if self.config.tke_budget_on or self.config.tke_production_on:
            self.statistics.append(TKE_Production())

        if self.config.tke_budget_on or self.config.tke_dissipation_on:
            self.statistics.append(TKE_Dissipation(Re))

        if self.config.tke_budget_on or self.config.tke_convection_on:
            self.statistics.append(TKE_Convection())

        if self.config.tke_budget_on or self.config.tke_viscous_diffusion_on:
            self.statistics.append(TKE_ViscousDiffusion(Re))

        if self.config.tke_budget_on or self.config.tke_pressure_transport_on:
            self.statistics.append(TKE_PressureTransport())

        if self.config.tke_budget_on or self.config.tke_turbulent_diffusion_on:
            self.statistics.append(TKE_TurbulentDiffusion())

    def compute_all(self) -> None:
        """Compute all registered statistics for all cases and timesteps"""
        total_tasks = len(self.statistics) * len(self.config.cases) * len(self.config.timesteps)
        with tqdm(total=total_tasks, desc="Computing statistics", unit="stat") as pbar:
            for stat in self.statistics:
                for case in self.config.cases:
                    for timestep in self.config.timesteps:
                        stat.compute_for_case(case, timestep, self.data_loader)
                        pbar.update(1)

    def process_all(self) -> None:
        """Apply normalization and averaging to all computed statistics"""
        total_tasks = sum(len(stat.raw_results) for stat in self.statistics)
        with tqdm(total=total_tasks, desc="Processing statistics", unit="stat") as pbar:
            for stat in self.statistics:
                for (case, timestep), values in stat.raw_results.items():

                    # Get u1 data for normalization
                    ux_data = self.data_loader.get(case, 'u1', timestep)
                    if ux_data is None:
                        print(f'Missing u1 data for normalization: {case}, {timestep}')
                        pbar.update(1)
                        continue

                    ref_Re = op.get_ref_Re(case, self.config.cases, self.config.Re)

                    # Normalize
                    if self.config.norm_by_u_tau_sq and stat.name != 'temperature':
                        normed = op.norm_turb_stat_wrt_u_tau_sq(ux_data, values, ref_Re)
                    else:
                        normed = values

                    # Special normalization for u1 velocity
                    if self.config.norm_ux_by_u_tau and stat.name == 'ux_velocity':
                        normed = op.norm_ux_velocity_wrt_u_tau(ux_data, ref_Re)
                        print(f'u1 velocity normalised by u_tau for {case}, {timestep}')

                    # Symmetric averaging (skip for u'v')
                    if self.config.half_channel_plot:
                        if stat.name != 'u_prime_v_prime' and stat.name != 'temperature':
                            normed_avg = op.symmetric_average(normed)
                            stat.processed_results[(case, timestep)] = normed_avg
                        else:
                            stat.processed_results[(case, timestep)] = normed[:(len(normed)//2)]
                    else:
                        stat.processed_results[(case, timestep)] = normed

                    # Print flow info
                    cur_Re = op.get_Re(case, self.config.cases, self.config.Re, ux_data, self.config.forcing)
                    ut.print_flow_info(ux_data, ref_Re, cur_Re, case, timestep)
                    pbar.update(1)

    def get_statistic(self, name: str) -> Optional[Union[ReStresses, Profiles, TkeBudget]]:
        """Get a specific statistic by name"""
        for stat in self.statistics:
            if stat.name == name:
                return stat
        return None

    def get_statistics_by_class(self) -> Dict[str, List[Union[ReStresses, Profiles, TkeBudget]]]:
        """Group statistics by their class type"""
        grouped: Dict[str, List[Union[ReStresses, Profiles, TkeBudget]]] = {
            'ReStresses': [],
            'Profiles': [],
            'TkeBudget': []
        }
        for stat in self.statistics:
            if isinstance(stat, ReStresses):
                grouped['ReStresses'].append(stat)
            elif isinstance(stat, Profiles):
                grouped['Profiles'].append(stat)
            elif isinstance(stat, TkeBudget):
                grouped['TkeBudget'].append(stat)
        return grouped

# =====================================================================================================================================================
# PLOTTING CLASS
# =====================================================================================================================================================

class TurbulencePlotter:
    """Handles all plotting logic for turbulence statistics"""

    def __init__(self, config: Config, plot_config: PlotConfig, data_loader: TurbulenceTextData):
        self.config = config
        self.plot_config = plot_config
        self.data_loader = data_loader

    def plot(self, statistics: List[Union[ReStresses, Profiles, TkeBudget]], reference_data: Optional[ReferenceData] = None):
        """Main plotting method - delegates to single or multi plot"""
        if len(statistics) == 1 or not self.config.multi_plot:
            return self._plot_single_figure(statistics, reference_data)
        else:
            return self._plot_multi_figure(statistics, reference_data)

    def plot_by_class(self, grouped_statistics: Dict[str, List[Union[ReStresses, Profiles, TkeBudget]]],
                      reference_data: Optional[ReferenceData] = None) -> Dict[str, Any]:
        """Create separate figures for each class type (ReStresses, Profiles, TkeBudget)"""
        figures = {}

        class_titles = {
            'ReStresses': 'Reynolds Stresses',
            'Profiles': 'Profiles',
            'TkeBudget': 'TKE Budget'
        }

        for class_name, stats_list in grouped_statistics.items():
            if not stats_list:
                continue

            fig = self._plot_class_figure(stats_list, class_titles[class_name], reference_data)
            figures[class_name] = fig

        return figures

    def _plot_class_figure(self, statistics: List[Union[ReStresses, Profiles, TkeBudget]],
                           title: str, reference_data: Optional[ReferenceData] = None):
        """Create a figure with subplots for a single class type"""
        n_stats = len(statistics)

        if n_stats == 1:
            # Single subplot
            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            fig.canvas.manager.set_window_title(title)
            axs = np.array([[ax]])
            nrows, ncols = 1, 1
        else:
            ncols = math.ceil(math.sqrt(n_stats))
            nrows = math.ceil(n_stats / ncols)

            fig, axs = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(15, 10),
                constrained_layout=True
            )
            fig.canvas.manager.set_window_title(title)

            # Ensure axs is always 2D array
            if nrows == 1 and ncols == 1:
                axs = np.array([[axs]])
            elif nrows == 1 or ncols == 1:
                axs = axs.reshape(nrows, ncols)

        # Plot each statistic
        for i, stat in enumerate(statistics):
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]

            for (case, timestep), values in stat.processed_results.items():
                # Get y coordinates
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                # Get plotting aesthetics
                color = self._get_color(case, stat.name)
                label = f'{case.replace("_", " = ")}'

                # Plot main data
                self._plot_line(ax, y_plus, values, label, color)

                # Plot reference data
                if reference_data:
                    self._plot_reference_data(ax, stat.name, case, reference_data)

                # Add log scale reference lines
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(ax, y_plus)

            # Set subplot properties
            ax.set_title(f'{stat.label}')
            ax.set_ylabel(f"Normalised {stat.name.replace('_', ' ')}")
            ax.grid(True)
            ax.legend(fontsize='small')

            if row == nrows - 1:
                ax.set_xlabel('$y^+$')

        # Hide unused subplots
        for i in range(n_stats, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axs[row, col].set_visible(False)

        return fig

    def _plot_single_figure(self, statistics: List[Union[ReStresses, Profiles, TkeBudget]],
                           reference_data: Optional[ReferenceData] = None):
        """Create a single combined plot for all statistics"""
        plt.figure(figsize=(10, 6))
        plt.gcf().canvas.manager.set_window_title('Turbulence Statistics')

        for stat in statistics:
            for (case, timestep), values in stat.processed_results.items():

                # Get y coordinates
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                # Get plotting aesthetics
                color = self._get_color(case, stat.name)
                label = f'{stat.label}, {case.replace("_", " = ")}'

                # Plot main data
                self._plot_line(plt, y_plus, values, label, color)

                # Plot reference data
                if reference_data:
                    self._plot_reference_data(plt, stat.name, case, reference_data)

                # Add log scale reference lines
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(plt, y_plus)

        plt.xlabel('$y^+$')
 
        if len(statistics) == 1:
            plt.ylabel(f"Normalised {statistics[0].name.replace('_', ' ')}")
        else:
            plt.ylabel("Normalised Reynolds Stresses")

        plt.legend()
        plt.grid(True)

        return plt.gcf()

    def _plot_multi_figure(self, statistics: List[Union[ReStresses, Profiles, TkeBudget]],
                          reference_data: Optional[ReferenceData] = None):
        """Create separate subplots for each statistic"""
        n_stats = len(statistics)
        ncols = math.ceil(math.sqrt(n_stats))
        nrows = math.ceil(n_stats / ncols)

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(15, 10),
            constrained_layout=True
        )
        fig.canvas.manager.set_window_title('Turbulence Statistics')

        # Ensure axs is always 2D array
        if nrows == 1 and ncols == 1:
            axs = np.array([[axs]])
        elif nrows == 1 or ncols == 1:
            axs = axs.reshape(nrows, ncols)

        # Plot each statistic
        for i, stat in enumerate(statistics):
            row = i // ncols
            col = i % ncols
            ax = axs[row, col]

            for (case, timestep), values in stat.processed_results.items():

                # Get y coordinates
                y_plus = self._get_y_plus(case, timestep)
                if y_plus is None:
                    continue

                # Get plotting aesthetics
                color = self._get_color(case, stat.name)
                label = f'{case.replace("_", " = ")}'

                # Plot main data
                self._plot_line(ax, y_plus, values, label, color)

                # Plot reference data
                if reference_data:
                    self._plot_reference_data(ax, stat.name, case, reference_data)

                # Add log scale reference lines
                if stat.name == 'ux_velocity' and self.config.ux_velocity_log_ref_on and self.config.log_y_scale:
                    self._plot_log_reference_lines(ax, y_plus)

            # Set subplot properties
            ax.set_title(f'{stat.label}')
            ax.set_ylabel(f"Normalised {stat.name.replace('_', ' ')}")
            ax.grid(True)
            ax.legend(fontsize='small')

            if row == nrows - 1:
                ax.set_xlabel('$y^+$')

        # Hide unused subplots
        for i in range(n_stats, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axs[row, col].set_visible(False)

        return fig

    def _plot_line(self, ax, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
        """Plot a single line with appropriate scale"""
        if self.config.log_y_scale:
            ax.semilogx(x, y, label=label, linestyle='-', marker='', color=color)
        elif self.config.linear_y_scale:
            ax.plot(x, y, label=label, linestyle='-', marker='', color=color)
        else:
            print('Plotting input incorrectly defined')

    def _plot_reference_data(self, ax, stat_name: str, case: str,
                            reference_data: ReferenceData) -> None:
        """Plot reference data for a given statistic"""
        # MKM180 reference
        if self.config.mkm180_ch_ref_on and reference_data.mkm180_stats:
            if stat_name in reference_data.mkm180_stats:
                self._plot_line(ax, reference_data.mkm180_y_plus,
                              reference_data.mkm180_stats[stat_name],
                              f'{self.plot_config.stat_labels[stat_name]} MKM180',
                              'black')

        # Noguchi & Kasagi reference
        if self.config.mhd_NK_ref_on:
            if case == 'Ha_4' and reference_data.NK_H4_stats and stat_name in reference_data.NK_H4_stats:
                if self.config.log_y_scale:
                    ax.semilogx(reference_data.NK_ref_y_H4,
                              reference_data.NK_H4_stats[stat_name],
                              linestyle='', marker='o',
                              label='Ha = 4, Noguchi & Kasagi',
                              color=self.plot_config.colors_1[stat_name], markevery=2)
                else:
                    ax.plot(reference_data.NK_ref_y_H4,
                          reference_data.NK_H4_stats[stat_name],
                          linestyle='', marker='o',
                          label='Ha = 4, Noguchi & Kasagi',
                          color=self.plot_config.colors_1[stat_name], markevery=2)

            elif case == 'Ha_6' and reference_data.NK_H6_stats and stat_name in reference_data.NK_H6_stats:
                if self.config.log_y_scale:
                    ax.semilogx(reference_data.NK_ref_y_H6,
                              reference_data.NK_H6_stats[stat_name],
                              linestyle='', marker='o',
                              label='Ha = 6, Noguchi & Kasagi',
                              color=self.plot_config.colors_2[stat_name], markevery=2)
                else:
                    ax.plot(reference_data.NK_ref_y_H6,
                          reference_data.NK_H6_stats[stat_name],
                          linestyle='', marker='o',
                          label='Ha = 6, Noguchi & Kasagi',
                          color=self.plot_config.colors_2[stat_name], markevery=2)

    def _plot_log_reference_lines(self, ax, y_plus: np.ndarray) -> None:
        """Plot reference lines for log-scale velocity plots"""
        if self.config.log_y_scale:
            ax.semilogx(y_plus[:15], y_plus[:15], '--', linewidth=1,
                       label='$u^+ = y^+$', color='black', alpha=0.5)
            u_plus_ref = 2.5 * np.log(y_plus) + 5.5
            ax.plot(y_plus, u_plus_ref, '--', linewidth=1,
                   label='$u^+ = 2.5ln(y^+) + 5.5$', color='black', alpha=0.5)

    def _get_y_plus(self, case: str, timestep: str) -> Optional[np.ndarray]:
        """Calculate y+ coordinates for a case"""
        ux_data = self.data_loader.get(case, 'u1', timestep)
        if ux_data is None:
            print(f'Missing u1 data for plotting: {case}, {timestep}')
            return None

        if self.config.half_channel_plot:
            y = (ux_data[:len(ux_data)//2, 1] + 1)
        else:
            y = ux_data[:, 1]

        if self.config.norm_y_to_y_plus:
            cur_Re = op.get_Re(case, self.config.cases, self.config.Re, ux_data, self.config.forcing)
            return op.norm_y_to_y_plus(y, ux_data, cur_Re)
        else:
            y_plus = y
            return y_plus

    def _get_color(self, case: str, stat_name: str) -> str:
        """Get colour for a case and statistic"""
        if len(self.config.cases) <= len(self.plot_config.colours) and stat_name in self.plot_config.stat_labels:
            colour_set = ut.get_col(case, self.config.cases, self.plot_config.colours)
            return colour_set[stat_name]
        else:
            return np.random.choice(list(mcolors.CSS4_COLORS.keys()))

    def save_figure(self, fig, suffix: str = '') -> None:
        """Save figure to file"""
        if self.config.plot_name:
            base_name = self.config.plot_name.rsplit('.', 1)[0]
            ext = self.config.plot_name.rsplit('.', 1)[1] if '.' in self.config.plot_name else 'png'
            filename = f'{base_name}{suffix}.{ext}' if suffix else self.config.plot_name
        else:
            filename = f'turb_stats_plot{suffix}.png' if suffix else 'turb_stats_plot.png'
        if self.config.save_to_path and self.config.folder_path:
            save_path = os.path.join(self.config.folder_path, filename)
            fig.savefig(save_path,
                        dpi=300,
                        bbox_inches='tight',
                        pad_inches=0.1,
                        facecolor='white',
                        edgecolor='none',
                        transparent=True,
                        orientation='landscape')
            print(f'Figure saved to {save_path}')
        fig.savefig(f'turb_stats_plots/{filename}',
                   dpi=300,
                   bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white',
                   edgecolor='none',
                   transparent=True,
                   orientation='landscape')
        print(f'Figure saved to turb_stats_plots/{filename}')

    def save_figures_by_class(self, figures: Dict[str, Any]) -> None:
        """Save multiple figures, one for each class type"""
        for class_name, fig in figures.items():
            suffix = f'_{class_name.lower()}'
            self.save_figure(fig, suffix)

    def display_figure(self) -> None:
        """Display figure"""
        print('Displaying figure...')
        plt.show()



# =====================================================================================================================================================
# MAIN EXECUTION
# =====================================================================================================================================================

def main():
    """Main execution function"""
    # Import configuration
    import config as config_module
    config = Config.from_module(config_module)

    print("="*120)
    print("TURBULENCE STATISTICS PROCESSING")
    print("="*120)

    # Load turbulence data using the appropriate loader
    print("\nLoading turbulence data...")
    data_loader = create_data_loader(config)
    data_loader.load_all()

    # Load reference data
    print("\nLoading reference data...")
    reference_data = ReferenceData(config)
    reference_data.load_all()

    # Compute statistics
    print("\nComputing turbulence statistics...")
    pipeline = TurbulenceStatsPipeline(config, data_loader)
    pipeline.compute_all()

    # Process statistics (normalize, average)
    print("\nProcessing statistics...")
    print("\nFlow info:\n")
    pipeline.process_all()

    # Plot results
    if config.display_fig or config.save_fig:
        print("\nGenerating plots...")
        plot_config = PlotConfig()
        plotter = TurbulencePlotter(config, plot_config, data_loader)

        # Get statistics grouped by class type
        grouped_stats = pipeline.get_statistics_by_class()

        # Create separate figures for each class
        figures = plotter.plot_by_class(grouped_stats, reference_data)

        if config.save_fig:
            plotter.save_figures_by_class(figures)

        if config.display_fig:
            plotter.display_figure()
    else:
        print('\nNo output option selected')

    print("\n" + "="*120)
    print("PROCESSING COMPLETE")
    print("="*120)


if __name__ == '__main__':
    main()
