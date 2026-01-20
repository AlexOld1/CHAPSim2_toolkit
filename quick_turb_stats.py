#!/usr/bin/env python3
"""
Quick Turbulence Statistics Plotter
Simplified script for plotting turbulence statistics from a single XDMF case.
Requires only numpy and matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import math
import mmap
import glob
from tqdm import tqdm

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

# Set to True to use memory-mapped file reading (can help on HPC/Lustre filesystems)
USE_MMAP_READ = False

# Sampling stride for x and z directions (1 = no sampling, 2 = every 2nd point, etc.)
# Higher values = faster loading but less averaging samples
SAMPLE_STRIDE_XZ = 1


def parse_xdmf_file(xdmf_path):
    """
    Parse XDMF file and extract data from associated binary files.
    Returns dictionary of variable names to numpy arrays and grid info.
    """
    if not os.path.isfile(xdmf_path):
        return {}, {}

    try:
        # Read file content first, then parse from string
        # This avoids ET.parse() holding file locks on Lustre filesystems
        with open(xdmf_path, 'r') as f:
            xml_content = f.read()
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Error parsing {xdmf_path}: {e}")
        return {}, {}
    except IOError as e:
        print(f"Error reading {xdmf_path}: {e}")
        return {}, {}

    arrays = {}
    grid_info = {}
    xdmf_dir = os.path.dirname(xdmf_path)

    # Find grid information
    for grid in root.iter('Grid'):
        # Get topology dimensions
        for topo in grid.iter('Topology'):
            dims_str = topo.get('Dimensions')
            if dims_str:
                dims = tuple(int(d) for d in dims_str.split())
                grid_info['node_dimensions'] = dims
                grid_info['cell_dimensions'] = tuple(d - 1 for d in dims)

        # Collect all data items to read for progress bar
        read_tasks = []

        # Get geometry (grid coordinates)
        for geom in grid.iter('Geometry'):
            geom_type = geom.get('GeometryType')
            if geom_type == 'VXVYVZ':
                data_items = list(geom.iter('DataItem'))
                coord_names = ['x', 'y', 'z']
                for i, data_item in enumerate(data_items[:3]):
                    read_tasks.append(('grid', f'grid_{coord_names[i]}', data_item))

        # Get attributes (flow variables)
        for attribute in grid.iter('Attribute'):
            name = attribute.get('Name')
            data_item = attribute.find('DataItem')
            if data_item is not None:
                read_tasks.append(('array', name, data_item))

        # Read all data items with progress bar
        xdmf_name = os.path.basename(xdmf_path)
        with tqdm(total=len(read_tasks), desc=f"  {xdmf_name}", unit="var", leave=False) as pbar:
            for task_type, name, data_item in read_tasks:
                pbar.set_postfix_str(name[:20])
                data = read_binary_data_item(data_item, xdmf_dir)
                if data is not None:
                    if task_type == 'grid':
                        grid_info[name] = data
                    else:
                        arrays[name] = data
                pbar.update(1)

    return arrays, grid_info


def read_binary_data_item(data_item, xdmf_dir):
    """
    Read binary data from a DataItem element.
    """
    format_type = data_item.get('Format', 'Binary')
    if format_type != 'Binary':
        print(f"Unsupported format: {format_type}")
        return None

    # Get data properties
    dims_str = data_item.get('Dimensions', '')
    dims = tuple(int(d) for d in dims_str.split()) if dims_str else None

    number_type = data_item.get('NumberType', 'Float')
    precision = int(data_item.get('Precision', '8'))
    seek = int(data_item.get('Seek', '0'))

    # Determine numpy dtype
    if number_type == 'Float':
        dtype = np.float32 if precision == 4 else np.float64
    elif number_type == 'Int':
        dtype = np.int32 if precision == 4 else np.int64
    else:
        dtype = np.float64

    # Get binary file path
    bin_path = data_item.text.strip() if data_item.text else None
    if bin_path is None:
        return None

    # Resolve relative path - look in ../1_data relative to xdmf_dir
    data_dir = os.path.normpath(os.path.join(xdmf_dir, '..', '1_data'))
    bin_filename = os.path.basename(bin_path)
    bin_path = os.path.join(data_dir, bin_filename)

    if not os.path.isfile(bin_path):
        print(f"Binary file not found: {bin_path}")
        return None

    try:
        # Calculate expected number of elements to read
        # This avoids np.fromfile reading until EOF, which can hang on Lustre/HPC filesystems
        itemsize = np.dtype(dtype).itemsize
        if dims:
            count = int(np.prod(dims))
        else:
            # Fallback: calculate from file size
            file_size = os.path.getsize(bin_path)
            count = (file_size - seek) // itemsize

        if USE_MMAP_READ:
            # Memory-mapped reading - can be more reliable on some HPC filesystems
            with open(bin_path, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                byte_count = count * itemsize
                data = np.frombuffer(mm[seek:seek + byte_count], dtype=dtype).copy()
                mm.close()
        else:
            # Standard reading with explicit count (fixes Lustre EOF hanging)
            with open(bin_path, 'rb') as f:
                f.seek(seek)
                data = np.fromfile(f, dtype=dtype, count=count)

        if dims:
            expected_size = int(np.prod(dims))
            if data.size >= expected_size:
                data = data[:expected_size].reshape(dims)
            elif data.size < expected_size:
                print(f"Warning: Data size mismatch for {bin_path} (got {data.size}, expected {expected_size})")

        # Apply sampling in x and z directions if enabled (dims are typically z, y, x)
        if SAMPLE_STRIDE_XZ > 1 and data is not None and len(data.shape) == 3:
            data = data[::SAMPLE_STRIDE_XZ, :, ::SAMPLE_STRIDE_XZ]

        return data

    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None


def load_xdmf_data(visu_folder, timestep):
    """
    Load all available XDMF data from the 2_visu folder.
    Returns dictionary of all variables and grid info.
    """
    file_patterns = [
        f'domain1_flow_{timestep}.xdmf',
        f'domain1_thermo_{timestep}.xdmf',
        f'domain1_mhd_{timestep}.xdmf',
        f'domain1_t_avg_flow_{timestep}.xdmf',
        f'domain1_t_avg_thermo_{timestep}.xdmf',
        f'domain1_t_avg_mhd_{timestep}.xdmf',
        f'domain1_tsp_avg_flow_{timestep}.xdmf',
        f'domain1_tsp_avg_thermo_{timestep}.xdmf',
        f'domain1_tsp_avg_mhd_{timestep}.xdmf',
        f'domain1_time_averaged_flow_{timestep}.xdmf', # old format
        f'domain1_time_averaged_thermo_{timestep}.xdmf',
        f'domain1_time_averaged_mhd_{timestep}.xdmf',
    ]

    all_data = {}
    grid_info = {}

    # Find which files exist
    existing_files = [(p, os.path.join(visu_folder, p)) for p in file_patterns
                      if os.path.isfile(os.path.join(visu_folder, p))]

    with tqdm(total=len(existing_files), desc="Loading XDMF files", unit="file") as pbar:
        for pattern, filepath in existing_files:
            pbar.set_postfix_str(pattern[:30])
            arrays, file_grid_info = parse_xdmf_file(filepath)

            # Store grid info from first file that has it
            if not grid_info and file_grid_info:
                grid_info = file_grid_info

            all_data.update(arrays)
            pbar.update(1)

    return all_data, grid_info


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
    Handles different naming conventions (u1/ux, uu11/uu, etc.)
    """
    results = {}

    # Map variable names - try different conventions
    def find_var(names):
        for name in names:
            if name in data:
                return extract_y_profile(data[name], grid_info)
            # Also check with time_averaged_ prefix
            prefixed = f'time_averaged_{name}'
            if prefixed in data:
                return extract_y_profile(data[prefixed], grid_info)
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
    T = find_var(['T', 'temperature', 'temp'])
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
    print("\nTip: Use Tab for path completion, ~ for home, $VAR for env variables")
    visu_folder = input("Path to 2_visu folder: ").strip()
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

    display_input = input("Display figures? (y/n) [y]: ").strip().lower()
    display_fig = display_input != 'n'

    return {
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
    global SAMPLE_STRIDE_XZ

    # Get user input
    config = get_user_input()
    if config is None:
        return

    # Set sampling stride from config
    SAMPLE_STRIDE_XZ = config['sample_stride']

    print("\n" + "=" * 60)
    print("Loading data...")
    if SAMPLE_STRIDE_XZ > 1:
        print(f"(Sampling every {SAMPLE_STRIDE_XZ} points in x/z for faster loading)")
    print("=" * 60)

    # Load XDMF data
    data, grid_info = load_xdmf_data(config['visu_folder'], config['timestep'])

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

    save_path = config['visu_folder'] if config['save_fig'] else None
    plot_results(results, y, config['half_channel'], save_path)

    if config['display_fig']:
        plt.show()

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
