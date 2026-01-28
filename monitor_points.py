#!/usr/bin/env python3
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
blk_files = ['domain1_monitor_metrics_history.log', 'domain1_monitor_change_history.log']

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

        # Create subplots for all variables
        num_subplots = 6 if thermo_on else 5
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots), sharex=True)

        # Subplot 0: u-velocity
        axes[0].plot(time, u, label='u-velocity', linewidth=1, color='C0')
        axes[0].set_ylabel('u-velocity')
        axes[0].legend()
        axes[0].grid()

        # Subplot 1: v-velocity
        axes[1].plot(time, v, label='v-velocity', linewidth=1, color='C1')
        axes[1].set_ylabel('v-velocity')
        axes[1].legend()
        axes[1].grid()

        # Subplot 2: w-velocity
        axes[2].plot(time, w, label='w-velocity', linewidth=1, color='C2')
        axes[2].set_ylabel('w-velocity')
        axes[2].legend()
        axes[2].grid()

        # Subplot 3: Pressure
        axes[3].plot(time, p, label='pressure', linewidth=1, color='C3')
        axes[3].set_ylabel('Pressure')
        axes[3].legend()
        axes[3].grid()

        # Subplot 4: Pressure Correction
        axes[4].plot(time, phi, label='press. corr.', linewidth=1, color='C4')
        axes[4].set_ylabel('Pressure Correction')
        axes[4].legend()
        axes[4].grid()

        if thermo_on:
            # Subplot 5: Temperature
            axes[5].plot(time, T, label='temperature', linewidth=1, color='C5')
            axes[5].set_ylabel('Temperature')
            axes[5].legend()
            axes[5].grid()
            axes[5].set_xlabel('Time')
        else:
            axes[4].set_xlabel('Time')

        fig.suptitle(f'{file} - Monitor Point Data', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".dat","_plot")}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f'Saved subplot figure for {file}')

if plt_bulk:
    for file in blk_files:
        blk_data = np.loadtxt(path+file, skiprows=2)
        
        blk_data = blk_data[::sample_factor] # sample data for plotting

        if file == 'domain1_monitor_metrics_history.log':
            time = blk_data[:,0]
            MKE = blk_data[:,1]
            qx = blk_data[:,2]
            if thermo_on:
                gx = blk_data[:,3]
                T = blk_data[:,4]
                h = blk_data[:,5]

            # Create subplots for bulk quantities
            num_subplots = 4 if thermo_on else 2
            fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3*num_subplots), sharex=True)

            # Subplot 0: Mean Kinetic Energy
            axes[0].plot(time, MKE, label='Mean Kinetic Energy', linewidth=1, color='C0')
            axes[0].set_ylabel('Mean Kinetic Energy')
            axes[0].legend()
            axes[0].grid()

            # Subplot 1: Bulk Velocity and Density * Bulk Velocity (on same plot)
            axes[1].plot(time, qx, label='Bulk Velocity', linewidth=1, color='C1')
            if thermo_on:
                axes[1].plot(time, gx, label='Density * Bulk Velocity', linewidth=1, color='C2')
            axes[1].set_ylabel('Velocity')
            axes[1].legend()
            axes[1].grid()

            if thermo_on:
                # Subplot 2: Bulk Temperature
                axes[2].plot(time, T, label='Bulk Temperature', linewidth=1, color='C3')
                axes[2].set_ylabel('Bulk Temperature')
                axes[2].legend()
                axes[2].grid()

                # Subplot 3: Bulk Enthalpy
                axes[3].plot(time, h, label='Bulk Enthalpy', linewidth=1, color='C4')
                axes[3].set_ylabel('Bulk Enthalpy')
                axes[3].legend()
                axes[3].grid()
                axes[3].set_xlabel('Time')
            else:
                axes[1].set_xlabel('Time')

            fig.suptitle('Bulk Quantities', fontsize=14)
            fig.tight_layout()
            fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".log","_plot")}.png', dpi=300)
            print(f'Saved metrics history plot for {file}')
        
        if file == 'domain1_monitor_change_history.log':
            time = blk_data[:,0]
            mass_cons = blk_data[:,1]
            mass_chng_rt = blk_data[:,4]
            KE_chng_rt = blk_data[:,5]

            # Create subplots for change history
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            # Subplot 0: Mass Conservation
            axes[0].plot(time, mass_cons, label='Mass Conservation', linewidth=1, color='C0')
            axes[0].set_ylabel('Mass Conservation')
            axes[0].legend()
            axes[0].grid()

            # Subplot 1: Mass Change Rate
            axes[1].plot(time, mass_chng_rt, label='Mass Change Rate', linewidth=1, color='C1')
            axes[1].set_ylabel('Mass Change Rate')
            axes[1].legend()
            axes[1].grid()

            # Subplot 2: Kinetic Energy Change Rate
            axes[2].plot(time, KE_chng_rt, label='Kinetic Energy Change Rate', linewidth=1, color='C2')
            axes[2].set_ylabel('KE Change Rate')
            axes[2].legend()
            axes[2].grid()
            axes[2].set_xlabel('Time')

            fig.suptitle('Change History', fontsize=14)
            fig.tight_layout()
            fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".log","_plot")}.png', dpi=300)
            print(f'Saved change history plot for {file}')

print('='*100)
print(f'All plots saved to: {path}')
print('='*100)
