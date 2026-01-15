# CHAPSim2_python_toolkit
A python post-processing and toolkit program based on NumPy, Matplotlib and PyVista for DNS solver CHAPSim2.

Install:

Dependencies are given in requirements.txt.
pip: Navigate to base directory and run 'pip install .'
conda: Navigate to base directory and run 'conda env create -f environment.yml' to create a conva environment for the program then 'conda activate chapsim2-toolkit' to use the environment.

Scripts:

turb_stats.py: Main post-processing script to provide velocity, temperature & Reynolds stress profiles from CHAPSim2 text file output. Input parameters, cases for comparison etc. on config.py file. Plots saved in turb_stats_plots/ and to file path.

monitor_points.py: Plotting for bulk and point monitors. Run the script in the directory containing monitor point files or specify a path to files.

thermal_BC_calc.py: Property functions for liquid metals in CHAPSim2, functionality to output NIST format data file, convert a given Grashof number to constant wall temperature difference or heat flux (channel flow), calculate Prandtl number. Interactive input.

visualise.py: 3D domain visualisation using PyVista. Under development, not currently functional.

Reference Data:

Isothermal channel (MKM180), square duct (KTH) reference data is provided as well as isothermal and heated MHD reference data (NK). All reference data is openly accessible from published sources. Copyright for reference datasets remains with the original authors/publishers. See individual data files for citations.