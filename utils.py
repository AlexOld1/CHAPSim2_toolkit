import os
import numpy as np
import xml.etree.ElementTree as ET
import mmap
from tqdm import tqdm

# =====================================================================================================================================================
# XDMF READING CONFIGURATION
# =====================================================================================================================================================

# Set to True to use memory-mapped file reading (can help on HPC/Lustre filesystems)
USE_MMAP_READ = False

# Set to True to load all variables, False to load only required ones (faster)
LOAD_ALL_VARS = False

# Variables required for Reynolds stress computation (base names without prefixes)
_BASE_REQUIRED_VARS = [
    # Velocities
    'u1', 'u2', 'u3', 'ux', 'uy', 'uz', 'qx_ccc', 'qy_ccc', 'qz_ccc',
    # Reynolds stresses
    'uu11', 'uu12', 'uu13', 'uu22', 'uu23', 'uu33', 'uu', 'uv', 'vv', 'ww', 'uxux', 'uxuy', 'uyuy', 'uzuz',
    # Triple correlations
    'uuu111', 'uuu112', 'uuu113', 'uuu122', 'uuu123', 'uuu133',
    'uuu222', 'uuu223', 'uuu233', 'uuu333',
    # Velocity gradients
    'du1dx', 'du1dy', 'du1dz', 'du2dx', 'du2dy', 'du2dz', 'du3dx', 'du3dy', 'du3dz',
    # Dissipation terms
    'dudu11', 'dudu12', 'dudu13', 'dudu22', 'dudu23', 'dudu33',
    # TKE gradients
    'dkdx', 'dkdy', 'dkdz', 'd2kdx2', 'd2kdy2', 'd2kdz2',
    # Pressure terms
    'pr', 'pru1', 'pru2', 'pru3', 'd_pu1p_dx', 'd_pu2p_dy', 'd_pu3p_dz',
    # Turbulent diffusion
    'd_uiuiu1_dx', 'd_uiuiu2_dy', 'd_uiuiu3_dz',
    # Temperature
    'T', 'temperature', 'temp', 'TT',
]

# Build full set including prefixed versions (tsp_avg_, t_avg_)
REQUIRED_VARS = set(_BASE_REQUIRED_VARS)
for prefix in ['tsp_avg_', 't_avg_']:
    REQUIRED_VARS.update(f'{prefix}{var}' for var in _BASE_REQUIRED_VARS)

# =====================================================================================================================================================
# TEXT DATA UTILITIES
# =====================================================================================================================================================

def data_filepath(folder_path, case, quantity, timestep):
    return f'{folder_path}{case}/1_data/domain1_tsp_avg_{quantity}_{timestep}.dat'

def load_ts_avg_data(data_filepath):
    try:
        return np.loadtxt(data_filepath)
    except OSError:
        print(f'Error loading data for {data_filepath}')
        return None

def get_quantities(thermo_on):
    quantities = ['u1', 'u2', 'u3', 'uu11', 'uu12', 'uu22','uu33','pr']
    if thermo_on:
        quantities.append('T')
    return quantities

# =====================================================================================================================================================
# XDMF FILE PATH UTILITIES
# =====================================================================================================================================================

def visu_file_paths(folder_path, case, timestep):
    file_names = [
        f'{folder_path}{case}/2_visu/domain1_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_t_avg_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_tsp_avg_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_thermo_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_t_avg_thermo_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_tsp_avg_thermo_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_mhd_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_t_avg_mhd_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_tsp_avg_mhd_{timestep}.xdmf',
    ]
    return file_names

# =====================================================================================================================================================
# XDMF READING UTILITIES
# =====================================================================================================================================================

def read_binary_data_item(data_item, xdmf_dir):
    """
    Read binary data from a DataItem element.

    Args:
        data_item: XML DataItem element
        xdmf_dir: Directory containing the XDMF file

    Returns:
        numpy array or None if reading fails
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

        return data

    except Exception as e:
        print(f"Error reading {bin_path}: {e}")
        return None

def parse_xdmf_file(xdmf_path, load_all_vars=None, output_dim=1):
    """
    Parse XDMF file and extract data from associated binary files.

    Args:
        xdmf_path: Path to the XDMF file
        load_all_vars: If True, load all variables. If False, only load required ones.
                       If None, uses module-level LOAD_ALL_VARS setting.
        output_dim: Desired output dimension (1, 2, or 3)

    Returns:
        tuple: (arrays dict, grid_info dict)
    """
    if load_all_vars is None:
        load_all_vars = LOAD_ALL_VARS

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

        # Collect all data items to read
        read_tasks = []

        # Get geometry (grid coordinates)
        for geom in grid.iter('Geometry'):
            geom_type = geom.get('GeometryType')
            if geom_type == 'VXVYVZ':
                data_items = list(geom.iter('DataItem'))
                coord_names = ['x', 'y', 'z']
                for i, data_item in enumerate(data_items[:3]):
                    read_tasks.append(('grid', f'grid_{coord_names[i]}', data_item))

        # Get attributes (flow variables) - filter to only required ones if enabled
        skipped_vars = []
        for attribute in grid.iter('Attribute'):
            name = attribute.get('Name')
            data_item = attribute.find('DataItem')
            if data_item is not None:
                if load_all_vars or name in REQUIRED_VARS:
                    read_tasks.append(('array', name, data_item))
                else:
                    skipped_vars.append(name)

        # Read all data items
        if skipped_vars:
            tqdm.write(f"  Skipping {len(skipped_vars)} unneeded variables: {', '.join(skipped_vars[:3])}{'...' if len(skipped_vars) > 3 else ''}")

        xdmf_name = os.path.basename(xdmf_path)
        for task_type, name, data_item in tqdm(read_tasks, desc=f"  Reading {xdmf_name}", unit="var", leave=False):
            data_3d = read_binary_data_item(data_item, xdmf_dir)
            if data_3d is not None:
                # Reshape flat arrays to 3D using topology dimensions if needed
                if len(data_3d.shape) == 1 and 'cell_dimensions' in grid_info:
                    cell_dims = grid_info['cell_dimensions']
                    if data_3d.size == int(np.prod(cell_dims)):
                        data_3d = data_3d.reshape(cell_dims)
                if output_dim == 1 and len(data_3d.shape) == 3:
                    data = data_3d[data_3d.shape[0]//2, :, data_3d.shape[2]//2].copy()
                    del data_3d
                else:
                    data = data_3d
                if task_type == 'grid':
                    grid_info[name] = data
                else:
                    arrays[name] = data

    return arrays, grid_info


def xdmf_reader_wrapper(file_names, case=None, timestep=None, load_all_vars=None, data_types=None):
    """
    Reads XDMF files and extracts numpy arrays from binary data.

    Args:
        file_names (list): List of XDMF file paths to read
        case (str, optional): Case identifier for dictionary key
        timestep (str, optional): Timestep identifier for dictionary key
        load_all_vars (bool, optional): If True, load all variables. If False, only required ones.
        data_types (list, optional): List of data types to load. If None, loads all files.
            Valid types: 'inst', 't_avg', 'tsp_avg',
            or specific combinations like 't_avg_flow', 'tsp_avg_thermo', etc.
            Examples:
                - ['tsp_avg'] loads only time-space averaged files (flow, thermo, mhd)
                - ['inst'] loads all instantaneous files (flow, thermo, mhd)
                - ['tsp_avg_flow'] loads only time-space averaged flow files

    Returns:
        tuple: (visu_arrays_dic dict, grid_info dict)

        If case and timestep are provided, returns nested dictionary:
        {
            "case_timestep": {
                "variable_name": numpy_array,
                ...
            }
        }

        Otherwise returns flat dictionary:
        {
            "variable_name": numpy_array,
            ...
        }
    """
    # Determine if we should use nested structure
    use_nested = case is not None and timestep is not None

    if use_nested:
        # Create the outer key for this case/timestep combination
        outer_key = f"{case}_{timestep}"
        visu_arrays_dic = {outer_key: {}}
        inner_dict = visu_arrays_dic[outer_key]
    else:
        visu_arrays_dic = {}
        inner_dict = visu_arrays_dic

    grid_info = {}

    # Filter to only existing files for accurate progress bar
    existing_files = [f for f in file_names if os.path.isfile(f)]

    # Filter by data types if specified
    if data_types is not None:
        filtered_files = []
        for f in existing_files:
            filename = os.path.basename(f)
            # Check if file matches any of the requested data types
            for dtype in data_types:
                if dtype in ['tsp_avg', 't_avg']:
                    # Match prefix patterns like 'tsp_avg_flow', 'tsp_avg_thermo', 'tsp_avg_mhd'
                    if f'{dtype}_' in filename:
                        filtered_files.append(f)
                        break
                elif dtype == 'inst':
                    # Match instantaneous files (flow, thermo, mhd without t_avg or tsp_avg prefix)
                    if 't_avg' not in filename and 'tsp_avg' not in filename:
                        filtered_files.append(f)
                        break
                else:
                    # Match specific combinations like 'tsp_avg_flow', 't_avg_thermo'
                    if dtype in filename:
                        filtered_files.append(f)
                        break
        existing_files = filtered_files
        if data_types:
            tqdm.write(f"Filtering for data types: {data_types}")

    # If both t_avg and tsp_avg are requested, process tsp_avg first and t_avg last
    # so t_avg values take precedence when prefixes are stripped.
    if data_types is not None and 't_avg' in data_types and 'tsp_avg' in data_types:
        existing_files.sort(key=lambda f: 1 if 't_avg_' in os.path.basename(f) else 0)

    for xdmf_file in tqdm(existing_files, desc="Processing XDMF files", unit="file"):
        try:
            tqdm.write(f"Opening file: {xdmf_file}")

            # Parse the XDMF file

            arrays, file_grid_info = parse_xdmf_file(xdmf_file, load_all_vars=load_all_vars, output_dim=1 if 'tsp_avg' in xdmf_file else 3)

            if arrays:
                # Extract file type from filename for prefixing (optional)
                if 'tsp_avg' in xdmf_file:
                    file_type = 'tsp_avg'
                elif 't_avg' in xdmf_file:
                    file_type = 't_avg'
                elif '_mhd_' in xdmf_file:
                    file_type = 'mhd'
                elif '_thermo_' in xdmf_file:
                    file_type = 'thermo'
                else:
                    file_type = 'flow'

                # Store grid info if not already stored
                if not grid_info and file_grid_info:
                    grid_info = file_grid_info
                    if 'node_dimensions' in grid_info:
                        tqdm.write(f"Grid info: node_dimensions={grid_info['node_dimensions']}, cell_dimensions={grid_info.get('cell_dimensions', 'N/A')}")

                # Add arrays to dictionary, stripping averaging prefixes for consistency
                for var_name, var_data in arrays.items():
                    # Strip tsp_avg_ or t_avg_ prefix so statistics code can use base names
                    if var_name.startswith('tsp_avg_'):
                        base_name = var_name[8:]  # Remove 'tsp_avg_'
                    elif var_name.startswith('t_avg_'):
                        base_name = var_name[6:]  # Remove 't_avg_'
                    else:
                        base_name = var_name
                    inner_dict[base_name] = var_data
                tqdm.write(f"Successfully extracted {len(arrays)} arrays from {file_type} file")
            else:
                tqdm.write(f"Warning: No valid output from {xdmf_file}, file missing or empty")

        except Exception as e:
            tqdm.write(f"Error processing {xdmf_file}: {str(e)}")
            continue

    return visu_arrays_dic, grid_info


def extract_grid_info_from_arrays(grid_info):
    """
    Extract grid dimensions and bounds from grid coordinate arrays.

    Args:
        grid_info: Dictionary containing grid_x, grid_y, grid_z arrays

    Returns:
        dict: Enhanced grid information including bounds
    """
    enhanced_info = dict(grid_info)

    try:
        if 'grid_x' in grid_info and 'grid_y' in grid_info and 'grid_z' in grid_info:
            x = grid_info['grid_x']
            y = grid_info['grid_y']
            z = grid_info['grid_z']

            enhanced_info['bounds'] = (
                float(x.min()), float(x.max()),
                float(y.min()), float(y.max()),
                float(z.min()), float(z.max())
            )

            # Calculate average spacing
            if len(x) > 1:
                dx = (x.max() - x.min()) / (len(x) - 1)
            else:
                dx = 0
            if len(y) > 1:
                dy = (y.max() - y.min()) / (len(y) - 1)
            else:
                dy = 0
            if len(z) > 1:
                dz = (z.max() - z.min()) / (len(z) - 1)
            else:
                dz = 0

            enhanced_info['average_spacing'] = (dx, dy, dz)

    except Exception as e:
        print(f"Could not extract complete grid info: {str(e)}")

    return enhanced_info

# =====================================================================================================================================================
# OUTPUT UTILITIES
# =====================================================================================================================================================

def reader_output_summary(arrays_dict):
    """
    Provides a summary analysis of the extracted arrays.

    Args:
        arrays_dict (dict): Dictionary of numpy arrays (can be nested or flat)
    """
    print("\n" + "="*60)
    print("READER OUTPUT SUMMARY")
    print("="*60)

    # Check if this is a nested dictionary (case_timestep structure)
    first_key = next(iter(arrays_dict))
    is_nested = isinstance(arrays_dict[first_key], dict)

    if is_nested:
        # Handle nested structure: {case_timestep: {variable: array}}
        for case_timestep, variables in arrays_dict.items():
            print(f"\n{case_timestep}:")
            print("-" * 60)
            for var_name, array in variables.items():
                print(f"  {var_name}:")
                print(f"    Shape: {array.shape},  Min: {np.min(array):.6e},  Max: {np.max(array):.6e},  Mean: {np.mean(array):.6e}")
    else:
        # Handle flat structure: {variable: array}
        for key, array in arrays_dict.items():
            print(f"{key}:")
            print(f"  Shape: {array.shape},  Min value: {np.min(array):.6e},  Max value: {np.max(array):.6e}   Mean value: {np.mean(array):.6e}")
            print("-" * 40)

# =====================================================================================================================================================
# DATA CLEANING UTILITIES
# =====================================================================================================================================================

def clean_dat_file(input_file, output_file, expected_cols):
    clean_data = []
    bad_lines = []

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line_num <= 3:
                continue
            try:
                values = [float(x) for x in line.split()]
                if len(values) == expected_cols:
                    clean_data.append(values)
                else:
                    bad_lines.append((line_num, len(values), line.strip()))

            except ValueError as e:
                bad_lines.append((line_num, 'ERROR', line.strip()))

    if bad_lines:
        print(f"Found {len(bad_lines)} problematic lines")

    np.savetxt(f'monitor_point_plots/{output_file}', clean_data, fmt='%.5E')
    print(f"\nSaved {len(clean_data)} clean lines to {output_file}")

    return np.array(clean_data)

# =====================================================================================================================================================
# PLOTTING UTILITIES
# =====================================================================================================================================================

def get_col(case, cases, colours):
    if len(cases) > 1:
        colour = colours[cases.index(case)]
    else:
        colour = colours[0]
    return colour

def print_flow_info(ux_data, Re_ref, Re_bulk, case, timestep):

    Re_ref = int(Re_ref)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    tau_w = dudy/Re_ref # this should be ref Re not real bulk Re
    u_tau = np.sqrt(abs(dudy/Re_ref))
    Re_tau = u_tau * Re_ref
    print(f'Case: {case}, Timestep: {timestep}')
    print(f'Re_bulk = {Re_bulk}, u_tau = {u_tau}, tau_w = {tau_w}, Re_tau = {Re_tau}')
    print('-'*120)
    return

def get_plane_data(domain_array, plane, index):
    if plane == 'xy':
        return domain_array[index, :, :]
    elif plane == 'xz':
        return domain_array[:, index, :]
    elif plane == 'yz':
        return domain_array[:, :, index]
    else:
        print(f"Error: Invalid plane '{plane}' specified. Use 'xy', 'xz', or 'yz'.")
    return
