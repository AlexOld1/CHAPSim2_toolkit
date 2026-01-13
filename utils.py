import os
import vtk
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
import pyvista as pv

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

def visu_file_paths(folder_path, case, timestep):
    file_names = [
        f'{folder_path}{case}/2_visu/domain1_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_t_avg_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_tsp_avg_flow_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_mhd_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_thermo_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_t_avg_thermo_{timestep}.xdmf',
        f'{folder_path}{case}/2_visu/domain1_tsp_avg_thermo_{timestep}.xdmf'
    ]
    return file_names

def read_xdmf_extract_numpy_arrays(file_names, case=None, timestep=None):
    """
    Reads XDMF files and extracts numpy arrays from VTK data.

    Args:
        file_names (list): List of XDMF file paths to read
        case (str, optional): Case identifier for dictionary key
        timestep (str, optional): Timestep identifier for dictionary key

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
            "file_type_cell_variable_name": numpy_array,
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

    for xdmf_file in file_names:
        try:
            print(f"Opening file: {xdmf_file}")

            # Create an XdmfReader
            reader = vtk.vtkXdmfReader()
            reader.SetFileName(xdmf_file)

            # Try to update the reader - catch XML parsing errors
            try:
                reader.Update()
                output = reader.GetOutput()
            except Exception as xml_error:
                print(f"XML parsing error in {xdmf_file}: {str(xml_error)}")
                continue

            if output and output.GetNumberOfCells() > 0:
                # Extract file type from filename for prefixing
                if 't_avg' in xdmf_file:
                    file_type = 'time_avg'
                elif '_mhd_' in xdmf_file:
                    file_type = 'mhd'
                elif '_thermo_' in xdmf_file:
                    file_type = 'thermo'
                else:
                    file_type = 'flow'

                # Extract grid information if not already done
                if not grid_info:
                    grid_info = extract_grid_info(output)
                    print(f"Grid info: {grid_info}")

                # Get arrays from the dataset
                dataset_arrays = get_vtk_arrays_with_numpy(output, file_type, grid_info)
                inner_dict.update(dataset_arrays)
                print(f"Successfully extracted {len(dataset_arrays)} arrays from {file_type} file")
            else:
                print(f"Warning: No valid output from {xdmf_file}, file missing or empty")

        except Exception as e:
            print(f"Error processing {xdmf_file}: {str(e)}")
            continue

    return visu_arrays_dic, grid_info

def extract_grid_info(dataset):
    """
    Extract grid dimensions and spacing information from VTK dataset.
    
    Args:
        dataset: VTK dataset object
    
    Returns:
        dict: Grid information including dimensions and spacing
    """
    grid_info = {}
    
    try:
        # For structured grids
        if hasattr(dataset, 'GetDimensions'):
            dims = dataset.GetDimensions()
            grid_info['node_dimensions'] = dims
            grid_info['cell_dimensions'] = (dims[0]-1, dims[1]-1, dims[2]-1)
            print(f"Node dimensions: {dims}")
            print(f"Cell dimensions: {grid_info['cell_dimensions']}")
            
        # Get bounds
        if hasattr(dataset, 'GetBounds'):
            bounds = dataset.GetBounds()
            grid_info['bounds'] = bounds
            print(f"Domain bounds: x=[{bounds[0]:.3f}, {bounds[1]:.3f}], "
                  f"y=[{bounds[2]:.3f}, {bounds[3]:.3f}], z=[{bounds[4]:.3f}, {bounds[5]:.3f}]")
        
        # Calculate average spacing if possible
        if 'node_dimensions' in grid_info and 'bounds' in grid_info:
            dx = (bounds[1] - bounds[0]) / (dims[0] - 1)
            dy = (bounds[3] - bounds[2]) / (dims[1] - 1)
            dz = (bounds[5] - bounds[4]) / (dims[2] - 1)
            grid_info['average_spacing'] = (dx, dy, dz)
            print(f"Average Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
            
    except Exception as e:
        print(f"Could not extract complete grid info: {str(e)}")
        
    return grid_info

def get_vtk_arrays_with_numpy(dataset, file_type="", grid_info=None):
    """
    Extracts VTK arrays and converts them to NumPy arrays.
    
    Args:
        dataset: VTK dataset object
        file_type (str): Type of file for prefixing array names
        grid_info (dict): Grid information for reshaping arrays
    
    Returns:
        dict: Dictionary of numpy arrays
    """
    arrays = {}
    
    # Process Cell Data
    if dataset.GetCellData() and dataset.GetCellData().GetNumberOfArrays() > 0:
        print(f"\nCell Data Arrays ({file_type}):")
        for i in range(dataset.GetCellData().GetNumberOfArrays()):
            array = dataset.GetCellData().GetArray(i)
            name = array.GetName()
            numpy_array = vtk_to_numpy(array)
            
            # Reshape to 3D if grid info available
            if grid_info and 'cell_dimensions' in grid_info:
                try:
                    dims = grid_info['cell_dimensions']
                    if numpy_array.size == dims[0] * dims[1] * dims[2]:
                        numpy_array = numpy_array.reshape(dims[2], dims[1], dims[0])  # VTK uses different ordering
                        print(f"  {name}: Shape - {numpy_array.shape} (3D), Type - {numpy_array.dtype}")
                    else:
                        print(f"  {name}: Shape - {numpy_array.shape} (1D - size mismatch), Type - {numpy_array.dtype}")
                except:
                    print(f"  {name}: Shape - {numpy_array.shape} (1D - reshape failed), Type - {numpy_array.dtype}")
            else:
                print(f"  {name}: Shape - {numpy_array.shape} (1D), Type - {numpy_array.dtype}")
            
            key = f"{file_type}_cell_{name}" if file_type else f"cell_{name}"
            arrays[key] = numpy_array
    
    return arrays

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

def visualise_domain_var(output, flow_var):

    pv_mesh = pv.wrap(output)

    if pv_mesh:

        pv_mesh.cell_data[flow_var] = flow_var
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, scalars="qx_velocity", show_edges=True, cmap="viridis")
        plotter.show()

    else:
                    print("Error: Could not wrap VTK output to PyVista mesh.")

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
        #for line_num, cols, content in bad_lines:
        #    print(f"  Line {line_num} ({cols} cols): {content[:80]}...")
    
    np.savetxt(f'monitor_point_plots/{output_file}', clean_data, fmt='%.5E')  # Save clean data
    print(f"\nSaved {len(clean_data)} clean lines to {output_file}")
    
    return np.array(clean_data)

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

