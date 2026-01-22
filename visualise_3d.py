# import libraries ------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import os
import pyvista as pv

# import modules --------------------------------------------------------------------------------------------------------------------------------------

import operations as op
import utils as ut
from config import *

# Input parameters in development

# 3D visualisation (under construction)
visualisation_on = False

#instant_ux_on = False
#instant_uy_on = False
#instant_uz_on = False
#instant_press_on = False
#instant_phi_on = False

# 3D visualisation data (.xdmf files)

def visualise_domain_var(output, flow_var):

    pv_mesh = pv.wrap(output)

    if pv_mesh:

        pv_mesh.cell_data[flow_var] = flow_var
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, scalars="qx_velocity", show_edges=True, cmap="viridis")
        plotter.show()

    else:
                    print("Error: Could not wrap VTK output to PyVista mesh.")

visu_data = {}
visu_data.clear()
grid_info = {}

if visualisation_on:
    for case in cases:
        for timestep in timesteps:

            print(f"\n{'-'*60}")
            print(f"Processing: {case}, {timestep}")
            print(f"{'-'*60}")

            file_names = ut.visu_file_paths(folder_path, case, timestep)

            existing_files = [file for file in file_names if os.path.isfile(file)]
            missing_files = [file for file in file_names if not os.path.isfile(file)]
            for file in missing_files:
                print(f'No .xdmf file found for {file}')

            if existing_files:
                # Pass case and timestep to create nested dictionary structure
                arrays, grid_info_cur = ut.read_xdmf_extract_numpy_arrays(file_names, case=case, timestep=timestep)

                if arrays:
                    # Update visu_data with nested dictionary
                    # arrays is now {'case_timestep': {'var1': array1, 'var2': array2, ...}}
                    visu_data.update(arrays)

                    # Store grid info (should be same for all timesteps)
                    if not grid_info and grid_info_cur:
                        grid_info = grid_info_cur

                    # Get the nested key to count arrays
                    nested_key = f"{case}_{timestep}"
                    num_arrays = len(arrays[nested_key]) if nested_key in arrays else 0
                    print(f"\nSuccessfully extracted {num_arrays} arrays from case {case}, timestep {timestep}")
                else:
                    print(f"No arrays extracted from timestep {timestep}")
            else:
                continue

if visu_data and visualisation_on:
    ut.reader_output_summary(visu_data)
        
    # Print grid information
    if grid_info:
        print(f"\n{'='*60}")
        print("GRID INFORMATION")
        print(f"{'='*60}")
        for key, value in grid_info.items():
             print(f"{key}: {value}")

    # Print Array Extraction info
    print(f"\nTotal arrays extracted: {len(visu_data)}")                
else:
    print("No arrays were successfully extracted.")

# Whole-domain visualisation
# Slice visualisation
# Iso-surface visualisation
# Streamlines visualisation