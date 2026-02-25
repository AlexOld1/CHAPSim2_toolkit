#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

plt.rcParams['agg.path.chunksize'] = 10000 # Configure matplotlib for better performance with large datasets
plt.rcParams['path.simplify_threshold'] = 1.0

# ====================================================================================================================================================
# Robust y-limit calculation (IQR-based) for diverged simulations
# ====================================================================================================================================================

def compute_robust_ylim(data, iqr_factor=3.0, padding=0.05):
    """
    Compute robust y-axis limits using the Interquartile Range (IQR) method.
    
    Unlike mean ± k*sigma, the median and quartiles are resistant to extreme
    outliers, so this correctly identifies the 'physical' data range even when
    the simulation has diverged to very large values.
    
    Parameters
    ----------
    data : array-like
        The data array to compute limits for.
    iqr_factor : float
        Multiplier for the IQR to define the fence (default 3.0).
        Larger values are more permissive (show more of the data range).
    padding : float
        Fractional padding added to the computed range for visual comfort.
    
    Returns
    -------
    (ymin, ymax) or None if no clipping is needed.
    """
    finite_data = data[np.isfinite(data)]
    if len(finite_data) == 0:
        return None
    
    q1 = np.percentile(finite_data, 25)
    q3 = np.percentile(finite_data, 75)
    iqr = q3 - q1
    
    # Handle near-constant data where IQR ≈ 0
    if iqr < 1e-15:
        return None  # No meaningful spread to clip
    
    lower_fence = q1 - iqr_factor * iqr
    upper_fence = q3 + iqr_factor * iqr
    
    data_min = np.min(finite_data)
    data_max = np.max(finite_data)
    
    # Only apply if actual data exceeds the fence (i.e. divergence detected)
    if data_min >= lower_fence and data_max <= upper_fence:
        return None  # Data is well-behaved, let matplotlib auto-scale
    
    # Clamp bounds to at least cover all non-outlier data
    ymin = max(data_min, lower_fence)
    ymax = min(data_max, upper_fence)
    
    # Add visual padding
    span = ymax - ymin
    ymin -= padding * span
    ymax += padding * span
    
    return (ymin, ymax)


def apply_robust_ylim(ax, data):
    """
    Apply robust y-limits to a matplotlib axis if divergence is detected.
    Adds a text annotation when limits are clipped.
    """
    limits = compute_robust_ylim(data)
    if limits is not None:
        ax.set_ylim(limits)
        ax.annotate('⚠ y-axis clipped (divergence detected)',
                     xy=(0.5, 1.0), xycoords='axes fraction',
                     ha='center', va='bottom', fontsize=8,
                     color='red', fontstyle='italic')


def add_stats_box(ax, data):
    """
    Add a small statistics text box to a subplot.
    Shows mean, std, min, max and median of the plotted data.
    """
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return
    stats_text = (
        f"mean: {np.mean(finite):.4g}\n"
        f"std:  {np.std(finite):.4g}\n"
        f"min:  {np.min(finite):.4g}\n"
        f"max:  {np.max(finite):.4g}\n"
        f"med:  {np.median(finite):.4g}"
    )
    props = dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')


def running_average(data, window):
    """
    Compute a centred running average using a uniform convolution kernel.
    Returns an array the same length as data (edges use shrinking windows).
    """
    if window <= 1:
        return data.copy()
    kernel = np.ones(window) / window
    # 'same' keeps output length equal to input; edge effects are minor
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plot_with_avg(ax, time, data, label, color, window):
    """
    Plot raw data and, if a running average window is set, overlay the
    smoothed curve.
    """
    ax.plot(time, data, label=label, linewidth=0.8, color=color, alpha=0.45)
    if window > 1:
        avg = running_average(data, window)
        ax.plot(time, avg, label=f'{label} (avg, n={window})',
                linewidth=1.4, color=color)


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
display_plots = get_yes_no("Display plots interactively?", 'n')
auto_ylim = get_yes_no("Auto-limit y-axis range for diverged data?", 'y')
avg_window = get_int("Running average window size (1 = off)", 50)

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
        plot_with_avg(axes[0], time, u, 'u-velocity', 'C0', avg_window)
        axes[0].set_ylabel('u-velocity')
        axes[0].legend()
        axes[0].grid()
        if auto_ylim: apply_robust_ylim(axes[0], u)
        add_stats_box(axes[0], u)

        # Subplot 1: v-velocity
        plot_with_avg(axes[1], time, v, 'v-velocity', 'C1', avg_window)
        axes[1].set_ylabel('v-velocity')
        axes[1].legend()
        axes[1].grid()
        if auto_ylim: apply_robust_ylim(axes[1], v)
        add_stats_box(axes[1], v)

        # Subplot 2: w-velocity
        plot_with_avg(axes[2], time, w, 'w-velocity', 'C2', avg_window)
        axes[2].set_ylabel('w-velocity')
        axes[2].legend()
        axes[2].grid()
        if auto_ylim: apply_robust_ylim(axes[2], w)
        add_stats_box(axes[2], w)

        # Subplot 3: Pressure
        plot_with_avg(axes[3], time, p, 'pressure', 'C3', avg_window)
        axes[3].set_ylabel('Pressure')
        axes[3].legend()
        axes[3].grid()
        if auto_ylim: apply_robust_ylim(axes[3], p)
        add_stats_box(axes[3], p)

        # Subplot 4: Pressure Correction
        plot_with_avg(axes[4], time, phi, 'press. corr.', 'C4', avg_window)
        axes[4].set_ylabel('Pressure Correction')
        axes[4].legend()
        axes[4].grid()
        if auto_ylim: apply_robust_ylim(axes[4], phi)
        add_stats_box(axes[4], phi)

        if thermo_on:
            # Subplot 5: Temperature
            plot_with_avg(axes[5], time, T, 'temperature', 'C5', avg_window)
            axes[5].set_ylabel('Temperature')
            axes[5].legend()
            axes[5].grid()
            if auto_ylim: apply_robust_ylim(axes[5], T)
            add_stats_box(axes[5], T)
            axes[5].set_xlabel('Time')
        else:
            axes[4].set_xlabel('Time')

        fig.suptitle(f'{file} - Monitor Point Data', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".dat","_plot")}.png', dpi=300, bbox_inches='tight')
        if display_plots:
            plt.show()
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
            plot_with_avg(axes[0], time, MKE, 'Mean Kinetic Energy', 'C0', avg_window)
            axes[0].set_ylabel('Mean Kinetic Energy')
            axes[0].legend()
            axes[0].grid()
            if auto_ylim: apply_robust_ylim(axes[0], MKE)
            add_stats_box(axes[0], MKE)

            # Subplot 1: Bulk Velocity and Density * Bulk Velocity (on same plot)
            plot_with_avg(axes[1], time, qx, 'Bulk Velocity', 'C1', avg_window)
            if thermo_on:
                plot_with_avg(axes[1], time, gx, 'Density * Bulk Velocity', 'C2', avg_window)
            axes[1].set_ylabel('Velocity')
            axes[1].legend()
            axes[1].grid()
            if auto_ylim:
                combined = np.concatenate([qx, gx]) if thermo_on else qx
                apply_robust_ylim(axes[1], combined)
            add_stats_box(axes[1], qx)

            if thermo_on:
                # Subplot 2: Bulk Temperature
                plot_with_avg(axes[2], time, T, 'Bulk Temperature', 'C3', avg_window)
                axes[2].set_ylabel('Bulk Temperature')
                axes[2].legend()
                axes[2].grid()
                if auto_ylim: apply_robust_ylim(axes[2], T)
                add_stats_box(axes[2], T)

                # Subplot 3: Bulk Enthalpy
                plot_with_avg(axes[3], time, h, 'Bulk Enthalpy', 'C4', avg_window)
                axes[3].set_ylabel('Bulk Enthalpy')
                axes[3].legend()
                axes[3].grid()
                if auto_ylim: apply_robust_ylim(axes[3], h)
                add_stats_box(axes[3], h)
                axes[3].set_xlabel('Time')
            else:
                axes[1].set_xlabel('Time')

            fig.suptitle('Bulk Quantities', fontsize=14)
            fig.tight_layout()
            fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".log","_plot")}.png', dpi=300, bbox_inches='tight')
            if display_plots:
                plt.show()
            plt.close(fig)
            print(f'Saved metrics history plot for {file}')
        
        if file == 'domain1_monitor_change_history.log':
            time = blk_data[:,0]
            mass_cons = blk_data[:,1]
            mass_chng_rt = blk_data[:,4]
            KE_chng_rt = blk_data[:,5]

            # Create subplots for change history
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

            # Subplot 0: Mass Conservation
            plot_with_avg(axes[0], time, mass_cons, 'Mass Conservation', 'C0', avg_window)
            axes[0].set_ylabel('Mass Conservation')
            axes[0].legend()
            axes[0].grid()
            if auto_ylim: apply_robust_ylim(axes[0], mass_cons)
            add_stats_box(axes[0], mass_cons)

            # Subplot 1: Mass Change Rate
            plot_with_avg(axes[1], time, mass_chng_rt, 'Mass Change Rate', 'C1', avg_window)
            axes[1].set_ylabel('Mass Change Rate')
            axes[1].legend()
            axes[1].grid()
            if auto_ylim: apply_robust_ylim(axes[1], mass_chng_rt)
            add_stats_box(axes[1], mass_chng_rt)

            # Subplot 2: Kinetic Energy Change Rate
            plot_with_avg(axes[2], time, KE_chng_rt, 'Kinetic Energy Change Rate', 'C2', avg_window)
            axes[2].set_ylabel('KE Change Rate')
            axes[2].legend()
            axes[2].grid()
            if auto_ylim: apply_robust_ylim(axes[2], KE_chng_rt)
            add_stats_box(axes[2], KE_chng_rt)
            axes[2].set_xlabel('Time')

            fig.suptitle('Change History', fontsize=14)
            fig.tight_layout()
            fig.savefig(f'{path}{file.replace("domain1_monitor_","").replace(".log","_plot")}.png', dpi=300, bbox_inches='tight')
            if display_plots:
                plt.show()
            plt.close(fig)
            print(f'Saved change history plot for {file}')

print('='*100)
print(f'All plots saved to: {path}')
print('='*100)
