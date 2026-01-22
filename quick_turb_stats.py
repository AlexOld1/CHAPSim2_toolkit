#!/usr/bin/env python3
"""
Quick Turbulence Statistics Plotter
Simplified script for plotting turbulence statistics from a single XDMF case.
Requires only numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import glob
from tqdm import tqdm

# Import the shared XDMF reader from utils
import utils as ut

# Enable tab completion for path input
try:
    import readline
    def path_completer(text, state):
        # Expand ~ and environment variables
        expanded = os.path.expanduser(os.path.expandvars(text))
        # Add wildcard for glob matching
        if os.path.isdir(expanded):
            pattern = os.path.join(expanded, '*')
        else:
            pattern = expanded + '*'
        matches = glob.glob(pattern)
        # Add trailing slash for directories
        matches = [m + '/' if os.path.isdir(m) else m for m in matches]
        try:
            return matches[state]
        except IndexError:
            return None
    readline.set_completer(path_completer)
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
except ImportError:
    pass  # readline not available on all platforms


def load_xdmf_data(visu_folder, timestep, sample_stride=1):
    """
    Load all available XDMF data from the 2_visu folder.
    Prioritises tsp_avg (time-space averaged) files, falls back to t_avg (time averaged) if not found.
    Returns dictionary of all variables, grid info, and whether data is already space-averaged.
    """
    # Preferred: time-space averaged files (already spatially averaged, 1D in y)
    tsp_avg_patterns = [
        f'domain1_tsp_avg_flow_{timestep}.xdmf',
        f'domain1_tsp_avg_thermo_{timestep}.xdmf',
        f'domain1_tsp_avg_mhd_{timestep}.xdmf',
    ]

    # Fallback: time averaged files (3D, need spatial averaging)
    t_avg_patterns = [
        f'domain1_t_avg_flow_{timestep}.xdmf',
        f'domain1_t_avg_thermo_{timestep}.xdmf',
        f'domain1_t_avg_mhd_{timestep}.xdmf',
    ]

    # Check which tsp_avg files exist
    tsp_avg_files = [(p, os.path.join(visu_folder, p)) for p in tsp_avg_patterns
                     if os.path.isfile(os.path.join(visu_folder, p))]

    is_space_averaged = False
    if tsp_avg_files:
        print("Found time-space averaged (tsp_avg) files - data is 1D, no spatial averaging needed")
        file_patterns = tsp_avg_patterns
        is_space_averaged = True
    else:
        # Fall back to t_avg files
        t_avg_files = [(p, os.path.join(visu_folder, p)) for p in t_avg_patterns
                       if os.path.isfile(os.path.join(visu_folder, p))]
        if t_avg_files:
            print("No tsp_avg files found, using time averaged (t_avg) files - will spatially average")
            file_patterns = t_avg_patterns
        else:
            print("Warning: No averaged data files found (tsp_avg or t_avg)")
            file_patterns = []

    all_data = {}
    grid_info = {}

    # Find which files exist
    existing_files = [(p, os.path.join(visu_folder, p)) for p in file_patterns
                      if os.path.isfile(os.path.join(visu_folder, p))]

    # Use utils.py's parse_xdmf_file for each file
    with tqdm(total=len(existing_files), desc="Loading XDMF files", unit="file") as pbar:
        for pattern, filepath in existing_files:
            pbar.set_postfix_str(pattern[:30])

            # Use the shared XDMF reader from utils
            # For tsp_avg data, don't apply sampling (already 1D)
            # For t_avg data, apply sampling in x/z
            stride = 1 if is_space_averaged else sample_stride
            arrays, file_grid_info = ut.parse_xdmf_file(filepath, load_all_vars=False, sample_stride=stride)

            # Store grid info from first file that has it
            if not grid_info and file_grid_info:
                grid_info = file_grid_info

            all_data.update(arrays)
            pbar.update(1)

    return all_data, grid_info, is_space_averaged


def extract_y_profile(data_3d, grid_info):
    """
    Extract 1D y-profile by averaging over x and z directions.
    Data shape is typically (nz, ny, nx) for cell-centered data.
    """
    if len(data_3d.shape) == 3:
        # Average over x (axis 2) and z (axis 0)
        profile = np.mean(np.mean(data_3d, axis=2), axis=0)
    elif len(data_3d.shape) == 1:
        profile = data_3d
    else:
        profile = data_3d.flatten()
    return profile


def compute_reynolds_stresses(data, grid_info):
    """
    Compute Reynolds stresses from time-space averaged data.
    Handles different naming conventions (u1/ux, uu11/uu, etc.) and prefixes (tsp_avg_, t_avg_)
    """
    results = {}

    # Map variable names - try different conventions and prefixes
    def find_var(base_names):
        # Try with prefixes first (tsp_avg_, t_avg_), then without
        prefixes = ['tsp_avg_', 't_avg_', '']
        for prefix in prefixes:
            for name in base_names:
                full_name = f'{prefix}{name}'
                if full_name in data:
                    return extract_y_profile(data[full_name], grid_info)
        return None

    # Velocity components
    u1 = find_var(['u1', 'ux', 'qx_ccc'])
    u2 = find_var(['u2', 'uy', 'qy_ccc'])
    u3 = find_var(['u3', 'uz', 'qz_ccc'])

    # Reynolds stress correlations
    uu11 = find_var(['uu11', 'uu', 'uxux'])
    uu12 = find_var(['uu12', 'uv', 'uxuy'])
    uu22 = find_var(['uu22', 'vv', 'uyuy'])
    uu33 = find_var(['uu33', 'ww', 'uzuz'])

    # Store velocity profile
    if u1 is not None:
        results['ux_velocity'] = u1

    # Compute Reynolds stresses: <u'u'> = <uu> - <u><u>
    if u1 is not None and uu11 is not None:
        results['u_prime_sq'] = uu11 - np.square(u1)

    if u1 is not None and u2 is not None and uu12 is not None:
        results['u_prime_v_prime'] = uu12 - u1 * u2

    if u2 is not None and uu22 is not None:
        results['v_prime_sq'] = uu22 - np.square(u2)

    if u3 is not None and uu33 is not None:
        results['w_prime_sq'] = uu33 - np.square(u3)

    # Compute TKE if all components available
    if all(k in results for k in ['u_prime_sq', 'v_prime_sq', 'w_prime_sq']):
        results['TKE'] = 0.5 * (results['u_prime_sq'] + results['v_prime_sq'] + results['w_prime_sq'])

    # Temperature if available
    T = find_var(['T', 'temperature', 'temp', 'TT'])
    if T is not None:
        results['temperature'] = T

    return results


def compute_normalisation(ux_profile, y_coords, Re_bulk):
    """
    Compute wall shear stress and friction velocity for normalisation.
    """
    # Wall gradient (assuming wall at index 0)
    du = ux_profile[0] - ux_profile[1]

    if y_coords is not None and len(y_coords) > 1:
        dy = abs(y_coords[0] - y_coords[1])
    else:
        dy = 2.0 / len(ux_profile)  # Fallback assuming domain height of 2

    dudy = du / dy
    tau_w = dudy / Re_bulk
    u_tau = np.sqrt(abs(tau_w))
    u_tau_sq = abs(tau_w)

    Re_tau = u_tau * Re_bulk

    return u_tau, u_tau_sq, Re_tau


def normalise_results(results, y_coords, Re_bulk, ref_temp=None):
    """
    Normalise all results by friction velocity.
    """
    if 'ux_velocity' not in results:
        print("Warning: Cannot normalise without ux_velocity data")
        return results, None, None

    ux = results['ux_velocity']
    u_tau, u_tau_sq, Re_tau = compute_normalisation(ux, y_coords, Re_bulk)

    print(f"u_tau = {u_tau:.6f}, u_tau^2 = {u_tau_sq:.6f}, Re_tau = {Re_tau:.2f}")

    normalised = {}

    with tqdm(total=len(results), desc="Normalising", unit="var") as pbar:
        for name, values in results.items():
            pbar.set_postfix_str(name)
            if name == 'ux_velocity':
                normalised[name] = values / u_tau
            elif name == 'temperature':
                if ref_temp is not None:
                    normalised[name] = values * ref_temp
                else:
                    normalised[name] = values
            else:
                normalised[name] = values / u_tau_sq
            pbar.update(1)

    return normalised, u_tau, Re_tau


def get_y_coordinates(grid_info, n_points):
    """
    Get y coordinates from grid info or generate default.
    """
    if 'grid_y' in grid_info:
        y = grid_info['grid_y']
        # Cell centers if node coordinates
        if len(y) == n_points + 1:
            y = 0.5 * (y[:-1] + y[1:])
        return y[:n_points] if len(y) >= n_points else y
    else:
        # Default: assume channel with y in [-1, 1]
        return np.linspace(-1, 1, n_points)


def plot_results(results, y, half_channel=False, save_path=None):
    """
    Create plots for profiles and Reynolds stresses.
    """
    # Separate into profiles and Reynolds stresses
    profiles = {}
    re_stresses = {}

    profile_keys = ['ux_velocity', 'temperature', 'TKE']
    re_stress_keys = ['u_prime_sq', 'u_prime_v_prime', 'v_prime_sq', 'w_prime_sq']

    for key in profile_keys:
        if key in results:
            profiles[key] = results[key]

    for key in re_stress_keys:
        if key in results:
            re_stresses[key] = results[key]

    # Plot profiles
    if profiles:
        n_profiles = len(profiles)
        ncols = min(n_profiles, 3)
        nrows = math.ceil(n_profiles / ncols)

        fig1, axs1 = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                   constrained_layout=True)
        fig1.canvas.manager.set_window_title('Profiles')

        if n_profiles == 1:
            axs1 = np.array([[axs1]])
        elif nrows == 1 or ncols == 1:
            axs1 = axs1.reshape(nrows, ncols)

        labels = {
            'ux_velocity': ('$U^+$', 'Streamwise Velocity'),
            'temperature': ('$T$', 'Temperature'),
            'TKE': ('$k^+$', 'Turbulent Kinetic Energy'),
        }

        for i, (key, values) in enumerate(profiles.items()):
            row, col = i // ncols, i % ncols
            ax = axs1[row, col]

            plot_y = y[:len(values)]
            if half_channel:
                plot_y = plot_y + 1  # Shift to [0, 2] range

            ax.plot(plot_y, values, 'b-', linewidth=1.5)
            ylabel, title = labels.get(key, (key, key))
            ax.set_xlabel('$y$')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_profiles, nrows * ncols):
            row, col = i // ncols, i % ncols
            axs1[row, col].set_visible(False)

        if save_path:
            fig1.savefig(os.path.join(save_path, 'profiles.png'), dpi=300, bbox_inches='tight')
            print(f"Saved profiles.png")

    # Plot Reynolds stresses
    if re_stresses:
        n_stresses = len(re_stresses)
        ncols = min(n_stresses, 2)
        nrows = math.ceil(n_stresses / ncols)

        fig2, axs2 = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                                   constrained_layout=True)
        fig2.canvas.manager.set_window_title('Reynolds Stresses')

        if n_stresses == 1:
            axs2 = np.array([[axs2]])
        elif nrows == 1 or ncols == 1:
            axs2 = axs2.reshape(nrows, ncols)

        labels = {
            'u_prime_sq': ("$\\langle u'u' \\rangle^+$", "$\\langle u'u' \\rangle$"),
            'u_prime_v_prime': ("$\\langle u'v' \\rangle^+$", "$\\langle u'v' \\rangle$"),
            'v_prime_sq': ("$\\langle v'v' \\rangle^+$", "$\\langle v'v' \\rangle$"),
            'w_prime_sq': ("$\\langle w'w' \\rangle^+$", "$\\langle w'w' \\rangle$"),
        }

        colors = {'u_prime_sq': 'r', 'u_prime_v_prime': 'g', 'v_prime_sq': 'm', 'w_prime_sq': 'c'}

        for i, (key, values) in enumerate(re_stresses.items()):
            row, col = i // ncols, i % ncols
            ax = axs2[row, col]

            plot_y = y[:len(values)]
            if half_channel:
                plot_y = plot_y + 1

            ax.plot(plot_y, values, color=colors.get(key, 'b'), linewidth=1.5)
            ylabel, title = labels.get(key, (key, key))
            ax.set_xlabel('$y$')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(n_stresses, nrows * ncols):
            row, col = i // ncols, i % ncols
            axs2[row, col].set_visible(False)

        if save_path:
            fig2.savefig(os.path.join(save_path, 'reynolds_stresses.png'), dpi=300, bbox_inches='tight')
            print(f"Saved reynolds_stresses.png")

    return fig1 if profiles else None, fig2 if re_stresses else None


def get_user_input():
    """
    Interactively get configuration from user.
    """
    print("=" * 60)
    print("Quick Turbulence Statistics Plotter")
    print("=" * 60)

    # Get visu folder path (supports tab completion, ~, and $ENV_VARS)
    print("\nTip: Use Tab for path completion")
    case_folder = input("Path to case folder: ").strip()
    case_folder = os.path.expanduser(os.path.expandvars(case_folder))
    visu_folder = os.path.join(case_folder, '2_visu')
    visu_folder = os.path.expanduser(os.path.expandvars(visu_folder))
    if not os.path.isdir(visu_folder):
        print(f"Error: Directory not found: {visu_folder}")
        return None

    # List available timesteps
    xdmf_files = [f for f in os.listdir(visu_folder) if f.endswith('.xdmf')]
    timesteps = set()
    for f in xdmf_files:
        # Extract timestep from filename (e.g., domain1_tsp_avg_flow_100000.xdmf)
        parts = f.replace('.xdmf', '').split('_')
        if parts:
            timesteps.add(parts[-1])

    if timesteps:
        print(f"\nAvailable timesteps: {sorted(timesteps)}")

    timestep = input("Timestep: ").strip()

    # Thermo options
    thermo_input = input("Thermo enabled? (y/n) [n]: ").strip().lower()
    thermo_on = thermo_input == 'y'

    ref_temp = None
    if thermo_on:
        ref_temp_input = input("Reference temperature (K) [300]: ").strip()
        ref_temp = float(ref_temp_input) if ref_temp_input else 300.0

    # MHD options
    mhd_input = input("MHD enabled? (y/n) [n]: ").strip().lower()
    mhd_on = mhd_input == 'y'

    # Flow parameters
    forcing = input("Forcing type (CMF/CPG) [CMF]: ").strip().upper()
    if forcing not in ['CMF', 'CPG']:
        forcing = 'CMF'

    re_input = input("Reynolds number (bulk) [5000]: ").strip()
    Re = float(re_input) if re_input else 5000.0

    # Sampling option for faster loading
    print("\nSampling reduces data in x/z directions (y kept full for profiles)")
    print("  1 = full data, 2 = 1/4 data, 4 = 1/16 data, 8 = 1/64 data")
    print("  Recommended: 4-8 for quick preview, 1-2 for final results")
    sample_input = input("Sample stride [4]: ").strip()
    sample_stride = int(sample_input) if sample_input else 4
    if sample_stride < 1:
        sample_stride = 1

    # Half channel plot
    half_input = input("Plot half channel? (y/n) [n]: ").strip().lower()
    half_channel = half_input == 'y'

    # Save options
    save_input = input("Save figures? (y/n) [y]: ").strip().lower()
    save_fig = save_input != 'n'

    display_input = input("Display figures? (y/n) [n]: ").strip().lower()
    display_fig = display_input != 'n'

    return {
        'case_folder': case_folder,
        'visu_folder': visu_folder,
        'timestep': timestep,
        'thermo_on': thermo_on,
        'mhd_on': mhd_on,
        'forcing': forcing,
        'Re': Re,
        'ref_temp': ref_temp,
        'sample_stride': sample_stride,
        'half_channel': half_channel,
        'save_fig': save_fig,
        'display_fig': display_fig,
    }


def main():
    """Main execution function."""
    # Get user input
    config = get_user_input()
    if config is None:
        return

    print("\n" + "=" * 60)
    print("Loading data...")
    print("(Loading only required variables for Reynolds stress computation)")
    print("=" * 60)

    # Check if tsp_avg files exist to determine if we need sampling
    visu_folder = config['visu_folder']
    timestep = config['timestep']
    tsp_avg_exists = os.path.isfile(os.path.join(visu_folder, f'domain1_tsp_avg_flow_{timestep}.xdmf'))

    # Only apply sampling for t_avg (3D) data, not tsp_avg (already 1D)
    sample_stride = 1 if tsp_avg_exists else config['sample_stride']
    if sample_stride > 1:
        print(f"(Sampling every {sample_stride} points in x/z for spatial averaging)")

    # Load XDMF data
    data, grid_info, is_space_averaged = load_xdmf_data(visu_folder, timestep, sample_stride)

    if not data:
        print("Error: No data loaded. Check file paths.")
        print("\nAvailable files in folder:")
        for f in os.listdir(config['visu_folder']):
            print(f"  {f}")
        return

    print(f"\nLoaded {len(data)} variables: {list(data.keys())}")

    if grid_info:
        print(f"Grid info: {grid_info.get('cell_dimensions', 'unknown')}")

    # Compute statistics
    print("\n" + "=" * 60)
    print("Computing statistics...")
    print("=" * 60)

    results = compute_reynolds_stresses(data, grid_info)

    if not results:
        print("Error: Could not compute any statistics from loaded data.")
        print("Available variables:", list(data.keys()))
        return

    print(f"Computed: {list(results.keys())}")

    # Get y coordinates
    n_points = len(next(iter(results.values())))
    y = get_y_coordinates(grid_info, n_points)

    # Normalise
    print("\nNormalising...")
    results, u_tau, Re_tau = normalise_results(results, y, config['Re'], config['ref_temp'])

    # Handle half channel
    if config['half_channel']:
        half_n = n_points // 2
        y = y[:half_n]
        for key in results:
            vals = results[key]
            if key == 'u_prime_v_prime' or key == 'temperature':
                results[key] = vals[:half_n]
            else:
                # Symmetric average for other quantities
                results[key] = (vals[:half_n] + vals[::-1][:half_n]) / 2

    # Plot
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    save_path = config['case_folder'] if config['save_fig'] else None
    plot_results(results, y, config['half_channel'], save_path)

    if config['display_fig']:
        plt.show()

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
