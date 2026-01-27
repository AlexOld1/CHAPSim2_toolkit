#!/usr/bin/env python3
"""
2D Slice Visualiser for CFD Data
Simple script for generating 2D slices from XDMF datasets.
Requires only numpy, matplotlib, and tqdm.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm

# Import the shared XDMF reader from utils
import utils as ut

# Enable tab completion for path input
try:
    import readline
    def path_completer(text, state):
        expanded = os.path.expanduser(os.path.expandvars(text))
        if os.path.isdir(expanded):
            pattern = os.path.join(expanded, '*')
        else:
            pattern = expanded + '*'
        matches = glob.glob(pattern)
        matches = [m + '/' if os.path.isdir(m) else m for m in matches]
        try:
            return matches[state]
        except IndexError:
            return None
    readline.set_completer(path_completer)
    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
except ImportError:
    pass


def get_available_timesteps(visu_folder):
    """Extract available timesteps from XDMF filenames."""
    xdmf_files = [f for f in os.listdir(visu_folder) if f.endswith('.xdmf')]
    timesteps = set()
    for f in xdmf_files:
        parts = f.replace('.xdmf', '').split('_')
        if parts:
            timesteps.add(parts[-1])
    return sorted(timesteps)


def get_available_variables(data):
    """List all available variables in the loaded data."""
    return sorted(data.keys())


def extract_slice(data_3d, plane, index, grid_info):
    """
    Extract a 2D slice from 3D data.

    Args:
        data_3d: 3D numpy array with shape (nz, ny, nx)
        plane: 'xy', 'xz', or 'yz'
        index: slice index in the third dimension
        grid_info: dictionary containing grid_x, grid_y, grid_z

    Returns:
        slice_data: 2D numpy array
        coord1: coordinates for first axis
        coord2: coordinates for second axis
        axis_labels: tuple of (xlabel, ylabel)
    """
    nz, ny, nx = data_3d.shape

    if plane == 'xy':
        # Slice at constant z
        if index >= nz:
            index = nz - 1
        slice_data = data_3d[index, :, :]
        coord1 = grid_info.get('grid_x', np.arange(nx))
        coord2 = grid_info.get('grid_y', np.arange(ny))
        axis_labels = ('$x$', '$y$')

    elif plane == 'xz':
        # Slice at constant y
        if index >= ny:
            index = ny - 1
        slice_data = data_3d[:, index, :]
        coord1 = grid_info.get('grid_x', np.arange(nx))
        coord2 = grid_info.get('grid_z', np.arange(nz))
        axis_labels = ('$x$', '$z$')

    elif plane == 'yz':
        # Slice at constant x
        if index >= nx:
            index = nx - 1
        slice_data = data_3d[:, :, index]
        coord1 = grid_info.get('grid_y', np.arange(ny))
        coord2 = grid_info.get('grid_z', np.arange(nz))
        axis_labels = ('$y$', '$z$')

    else:
        raise ValueError(f"Invalid plane '{plane}'. Use 'xy', 'xz', or 'yz'.")

    return slice_data, coord1, coord2, axis_labels


def get_slice_location(grid_info, plane, index):
    """Get the physical location of the slice."""
    if plane == 'xy':
        coord_key = 'grid_z'
    elif plane == 'xz':
        coord_key = 'grid_y'
    elif plane == 'yz':
        coord_key = 'grid_x'
    else:
        return None

    if coord_key in grid_info:
        coords = grid_info[coord_key]
        if index < len(coords):
            return coords[index]
    return index


def plot_slice(slice_data, coord1, coord2, axis_labels, variable_name,
               cmap='viridis', vmin=None, vmax=None, symmetric=False,
               slice_info="", save_path=None, display=True):
    """
    Plot a 2D slice with colorbar.

    Args:
        slice_data: 2D numpy array
        coord1, coord2: coordinate arrays for axes
        axis_labels: tuple of (xlabel, ylabel)
        variable_name: name of the variable being plotted
        cmap: colormap name
        vmin, vmax: color scale limits
        symmetric: if True, use symmetric color scale around zero
        slice_info: string describing slice location
        save_path: path to save figure (None to skip saving)
        display: whether to display the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Handle color scale
    if symmetric:
        max_abs = max(abs(np.nanmin(slice_data)), abs(np.nanmax(slice_data)))
        vmin = -max_abs
        vmax = max_abs
        if cmap == 'viridis':
            cmap = 'RdBu_r'

    if vmin is None:
        vmin = np.nanmin(slice_data)
    if vmax is None:
        vmax = np.nanmax(slice_data)

    # Create meshgrid for pcolormesh
    if len(coord1) == slice_data.shape[1]:
        # Cell-centered data, need to create cell edges
        dx = np.diff(coord1)
        x_edges = np.concatenate([[coord1[0] - dx[0]/2],
                                   coord1[:-1] + dx/2,
                                   [coord1[-1] + dx[-1]/2]])
    else:
        x_edges = coord1

    if len(coord2) == slice_data.shape[0]:
        dy = np.diff(coord2)
        y_edges = np.concatenate([[coord2[0] - dy[0]/2],
                                   coord2[:-1] + dy/2,
                                   [coord2[-1] + dy[-1]/2]])
    else:
        y_edges = coord2

    # Plot
    pcm = ax.pcolormesh(x_edges, y_edges, slice_data,
                        cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')

    cbar = fig.colorbar(pcm, ax=ax, label=variable_name)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(f'{variable_name} {slice_info}')
    ax.set_aspect('equal', adjustable='box')

    # Print statistics
    print(f"\nSlice statistics for {variable_name}:")
    print(f"  Min: {np.nanmin(slice_data):.6e}")
    print(f"  Max: {np.nanmax(slice_data):.6e}")
    print(f"  Mean: {np.nanmean(slice_data):.6e}")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)

    return fig


def get_user_input():
    """Interactively get configuration from user."""
    print("=" * 60)
    print("2D Slice Visualiser")
    print("=" * 60)

    # Get case folder path
    print("\nTip: Use Tab for path completion, leave empty for current directory")
    case_folder = input("Path to case folder: ").strip()
    if not case_folder:
        case_folder = os.getcwd()
    case_folder = os.path.expanduser(os.path.expandvars(case_folder))

    # Handle case where user navigated to 2_visu folder
    if os.path.basename(case_folder) == '2_visu':
        visu_folder = case_folder
        case_folder = os.path.dirname(case_folder)
    else:
        visu_folder = os.path.join(case_folder, '2_visu')

    if not os.path.isdir(visu_folder):
        print(f"Error: Directory not found: {visu_folder}")
        return None

    # List available timesteps
    timesteps = get_available_timesteps(visu_folder)
    if timesteps:
        print(f"\nAvailable timesteps: {timesteps}")

    timestep = input("Timestep: ").strip()

    # Select data type
    print("\nData types:")
    print("  1. inst   - Instantaneous data")
    print("  2. t_avg  - Time-averaged data")
    data_type_choice = input("Data type [1]: ").strip()

    data_type_map = {'1': 'inst', '2': 't_avg',
                     'inst': 'inst', 't_avg': 't_avg',
                     '': 'inst'}
    data_type = data_type_map.get(data_type_choice, 'inst')

    return {
        'case_folder': case_folder,
        'visu_folder': visu_folder,
        'timestep': timestep,
        'data_type': data_type,
    }


def get_slice_config(data, grid_info):
    """Get slice configuration from user after data is loaded."""

    # List available variables
    variables = get_available_variables(data)
    print(f"\nAvailable variables ({len(variables)}):")
    for i, var in enumerate(variables):
        shape = data[var].shape
        print(f"  {i+1:2d}. {var:20s} shape: {shape}")

    var_choice = input("\nVariable to plot (name or number): ").strip()

    # Parse variable choice
    if var_choice.isdigit():
        idx = int(var_choice) - 1
        if 0 <= idx < len(variables):
            variable = variables[idx]
        else:
            print("Invalid index, using first variable")
            variable = variables[0]
    elif var_choice in variables:
        variable = var_choice
    else:
        print(f"Variable '{var_choice}' not found, using first variable")
        variable = variables[0]

    # Check if data is 3D
    var_data = data[variable]
    if len(var_data.shape) != 3:
        print(f"Error: Variable '{variable}' is not 3D (shape: {var_data.shape})")
        print("2D slicing requires 3D data.")
        return None

    nz, ny, nx = var_data.shape
    print(f"\nSelected: {variable} with shape (nz={nz}, ny={ny}, nx={nx})")

    # Select slice plane
    print("\nSlice planes:")
    print(f"  xy - constant z (0 to {nz-1})")
    print(f"  xz - constant y (0 to {ny-1})")
    print(f"  yz - constant x (0 to {nx-1})")

    plane = input("Slice plane [xy]: ").strip().lower()
    if plane not in ['xy', 'xz', 'yz']:
        plane = 'xy'

    # Get index range for selected plane
    if plane == 'xy':
        max_idx = nz - 1
        coord_name = 'z'
        coords = grid_info.get('grid_z', np.arange(nz))
    elif plane == 'xz':
        max_idx = ny - 1
        coord_name = 'y'
        coords = grid_info.get('grid_y', np.arange(ny))
    else:  # yz
        max_idx = nx - 1
        coord_name = 'x'
        coords = grid_info.get('grid_x', np.arange(nx))

    print(f"\n{coord_name} range: {coords.min():.4f} to {coords.max():.4f} (indices 0 to {max_idx})")

    # Get slice index or location
    idx_input = input(f"Slice index or {coord_name} value [{max_idx//2}]: ").strip()

    if idx_input == '':
        index = max_idx // 2
    else:
        try:
            val = float(idx_input)
            if val <= max_idx and val == int(val):
                # Interpret as index
                index = int(val)
            else:
                # Interpret as coordinate value, find nearest index
                index = np.argmin(np.abs(coords - val))
                print(f"  -> Nearest index: {index} ({coord_name} = {coords[index]:.4f})")
        except ValueError:
            index = max_idx // 2

    index = max(0, min(index, max_idx))

    # Colormap selection
    print("\nColormaps: viridis, plasma, inferno, magma, cividis, RdBu_r, coolwarm, jet")
    cmap = input("Colormap [viridis]: ").strip()
    if not cmap:
        cmap = 'viridis'

    # Color scale options
    print("\nColor scale options:")
    print("  1. Auto (min to max)")
    print("  2. Symmetric around zero")
    print("  3. Custom range")
    scale_choice = input("Scale option [1]: ").strip()

    symmetric = False
    vmin, vmax = None, None

    if scale_choice == '2':
        symmetric = True
    elif scale_choice == '3':
        vmin_input = input("vmin: ").strip()
        vmax_input = input("vmax: ").strip()
        vmin = float(vmin_input) if vmin_input else None
        vmax = float(vmax_input) if vmax_input else None

    # Save options
    save_input = input("\nSave figure? (y/n) [y]: ").strip().lower()
    save_fig = save_input != 'n'

    save_path = None
    if save_fig:
        default_name = f"{variable}_{plane}_slice.png"
        save_name = input(f"Filename [{default_name}]: ").strip()
        if not save_name:
            save_name = default_name
        save_path = input(f"Save directory [{os.getcwd()}]: ").strip()
        if not save_path:
            save_path = os.getcwd()
        save_path = os.path.join(os.path.expanduser(os.path.expandvars(save_path)), save_name)

    display_input = input("Display figure? (y/n) [y]: ").strip().lower()
    display = display_input != 'n'

    return {
        'variable': variable,
        'plane': plane,
        'index': index,
        'cmap': cmap,
        'symmetric': symmetric,
        'vmin': vmin,
        'vmax': vmax,
        'save_path': save_path,
        'display': display,
    }


def main():
    """Main execution function."""

    # Get initial configuration
    config = get_user_input()
    if config is None:
        return

    print("\n" + "=" * 60)
    print("Loading data...")
    print("=" * 60)

    # Build file list based on data type
    file_names = ut.visu_file_paths(
        config['case_folder'] + '/',
        '',
        config['timestep']
    )

    # Load data using wrapper
    data, grid_info = ut.xdmf_reader_wrapper(
        file_names,
        load_all_vars=True,
        data_types=[config['data_type']]
    )

    if not data:
        print("Error: No data loaded. Check file paths and timestep.")
        return

    print(f"\nLoaded {len(data)} variables")

    if grid_info:
        dims = grid_info.get('cell_dimensions', grid_info.get('node_dimensions', 'unknown'))
        print(f"Grid dimensions: {dims}")

    # Get slice configuration
    slice_config = get_slice_config(data, grid_info)
    if slice_config is None:
        return

    print("\n" + "=" * 60)
    print("Generating slice...")
    print("=" * 60)

    # Extract slice
    var_data = data[slice_config['variable']]
    slice_data, coord1, coord2, axis_labels = extract_slice(
        var_data,
        slice_config['plane'],
        slice_config['index'],
        grid_info
    )

    # Get slice location info
    slice_loc = get_slice_location(grid_info, slice_config['plane'], slice_config['index'])
    if slice_config['plane'] == 'xy':
        slice_info = f"(z = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(z index = {slice_loc})"
    elif slice_config['plane'] == 'xz':
        slice_info = f"(y = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(y index = {slice_loc})"
    else:
        slice_info = f"(x = {slice_loc:.4f})" if isinstance(slice_loc, float) else f"(x index = {slice_loc})"

    # Plot
    plot_slice(
        slice_data, coord1, coord2, axis_labels,
        slice_config['variable'],
        cmap=slice_config['cmap'],
        vmin=slice_config['vmin'],
        vmax=slice_config['vmax'],
        symmetric=slice_config['symmetric'],
        slice_info=slice_info,
        save_path=slice_config['save_path'],
        display=slice_config['display']
    )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
