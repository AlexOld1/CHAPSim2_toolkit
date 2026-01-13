import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams['agg.path.chunksize'] = 10000 # Configure matplotlib for better performance with large datasets
plt.rcParams['path.simplify_threshold'] = 1.0

# ====================================================================================================================================================
# Input parameters
# ====================================================================================================================================================

# Parse command line arguments for data path
if len(sys.argv) > 1:
    path = sys.argv[1]
    # Ensure path ends with trailing slash
    if not path.endswith('/'):
        path += '/'
else:
    # If no argument provided, use current working directory
    path = os.getcwd() + '/'

print('='*100)
print(f'Plotting monitor points from: {path}')
print('='*100)

# Interactive configuration prompts
def get_yes_no(prompt, default='y'):
    """Get yes/no input from user."""
    response = input(f"{prompt} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
    if response == '':
        return default == 'y'
    return response in ['y', 'yes']

def get_int(prompt, default):
    """Get integer input from user."""
    response = input(f"{prompt} [{default}]: ").strip()
    if response == '':
        return default
    try:
        return int(response)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default

# Get configuration from user
print("\nConfiguration:")
print("-" * 100)

num_monitor_pts = get_int("Number of monitor points to plot", 5)
thermo_on = get_yes_no("Include temperature data?", 'y')
sample_factor = get_int("Sample factor (plot every nth point)", 10)
plt_pts = get_yes_no("Plot monitor points?", 'y')
plt_bulk = get_yes_no("Plot bulk/change history?", 'y')

print("-" * 100)
print()

# Generate file lists based on number of monitor points
pt_files = [f'domain1_monitor_pt{i}_flow.dat' for i in range(1, num_monitor_pts + 1)]
blk_files = ['domain1_monitor_bulk_history.log', 'domain1_monitor_change_history.log']

# ====================================================================================================================================================
 
if plt_pts:
    for file in pt_files:
        
        data = np.loadtxt(path+file,skiprows=3)
        
        data = data[::sample_factor] # sample data for plotting
        print(f'Plotting {len(data)} points for {file}...')

        time = data[:,0]
        u = data[:,1]
        v = data[:,2]
        w = data[:,3]
        p = data[:,4]
        phi = data[:,5]
        if thermo_on:
            T = data[:,6]

        # Plot 1: Velocities and Temperature
        fig = plt.figure(figsize=(10,6))
        plt.plot(time, u, label='u-velocity', linewidth=0.5)
        plt.plot(time, v, label='v-velocity', linewidth=0.5)
        plt.plot(time, w, label='w-velocity', linewidth=0.5)
        if thermo_on:
            plt.plot(time, T, label='temperature', linewidth=0.5)
        plt.xlabel('Time')
        plt.ylabel('Velocity / Temperature')
        plt.title(f'{file} - Velocity')
        plt.legend()
        plt.grid()
        fig.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.dat','_velocity_temp_plot')}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Plot 2: Pressure
        fig = plt.figure(figsize=(10,6))
        plt.plot(time, p, label='pressure', linewidth=0.5, color='C3')
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        plt.title(f'{file} - Pressure')
        plt.legend()
        plt.grid()
        fig.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.dat','_pressure_plot')}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Plot 3: Pressure Correction
        fig = plt.figure(figsize=(10,6))
        plt.plot(time, phi, label='press. corr.', linewidth=0.5, color='C4')
        plt.xlabel('Time')
        plt.ylabel('Pressure Correction')
        plt.title(f'{file} - Pressure Correction')
        plt.legend()
        plt.grid()
        fig.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.dat','_pressure_corr_plot')}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'Saved 3 plots for {file}')

if plt_bulk:
    for file in blk_files:
        blk_data = np.loadtxt(path+file, skiprows=2)
        
        blk_data = blk_data[::sample_factor] # sample data for plotting

        if file == 'domain1_monitor_bulk_history.log':
            time = blk_data[:,0]
            MKE = blk_data[:,1]
            qx = blk_data[:,2]
            if thermo_on:
                gx = blk_data[:,3]
                T = blk_data[:,4]
                h = blk_data[:,5]

            plt.figure(figsize=(10,6))
            plt.plot(time, MKE, label='Mean Kinetic Energy', linewidth=0.5)
            plt.plot(time, qx, label='Bulk Velocity', linewidth=0.5)
            if thermo_on:
                plt.plot(time, T, label='Bulk Temperature', linewidth=0.5)
                plt.plot(time, h, label='Bulk Enthalpy', linewidth=0.5)
                plt.plot(time, gx, label='Density * Bulk Velocity', linewidth=0.5)
            plt.xlabel('Time')
            plt.ylabel('Bulk Flow Variables')
            plt.title('Bulk Quantities')
            plt.legend()
            plt.grid()
            plt.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.log','_plot')}.png', dpi=300)
            print(f'Saved bulk history plot for {file}')
        
        if file == 'domain1_monitor_change_history.log':
            time = blk_data[:,0]
            mass_cons = blk_data[:,1]
            mass_chng_rt = blk_data[:,4]
            KE_chng_rt = blk_data[:,5]

            plt.figure(figsize=(10,6))
            plt.plot(time, mass_cons, label='Mass Conservation', linewidth=0.5)
            #plt.plot(time, mass_chng_rt, label='Mass Change Rate', linewidth=0.5)
            #plt.plot(time, KE_chng_rt, label='Kinetic Energy Change Rate', linewidth=0.5)
            plt.xlabel('Time')
            plt.ylabel('Change History Variables')
            plt.title('Change History')
            plt.legend()
            plt.grid()
            plt.savefig(f'{path}{file.replace('domain1_monitor_','').replace('.log','_plot')}.png', dpi=300)
            print(f'Saved change history plot for {file}')

print('='*100)
print(f'All plots saved to: {path}')
print('='*100)
