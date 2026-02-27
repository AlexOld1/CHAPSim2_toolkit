# Configuration file for turb_stats script ============================================================================================================

# Define input cases ==================================================================================================================================

# format: folder_path/case/1_data/quantity_timestep.dat
#folder_path = '/home/alex/sim_results/mhd_channel_validation/CPG/'
folder_path = ''
input_format = 'visu' # 'text' (.dat) or 'visu' (.xdmf)
cases = ['Tests'] # case names must match folder names exactly
timesteps = ['680000']

thermo_on = True    
mhd_on = True

forcing = 'CMF' # 'CMF' or 'CPG'
Re = [5000] # indexing matches 'cases' if different Re used for different cases. Use bulk reference value for CPG.
ref_temp = [670] # Kelvin

# Output ==============================================================================================================================================

# Profiles
ux_velocity_on = True
temp_on = True
tke_on = True
profile_direction = 'y' # 'y' (wall-normal), 'x' (streamwise), or 'both'
slice_coords = '' # y-profiles: x coords for slices, e.g. '0.5,1.0' (blank = streamwise avg)
x_profile_y_coords = '' # x-profiles: y coords for slices, e.g. '0.0,0.5' (blank = channel centreline)
surface_plot_on = False # Plot 2D (y,x) surface contour maps of each statistic (requires 2D data, i.e. average_z_direction=True, average_x_direction=False)

# Reynolds stresses
u_prime_sq_on = True
u_prime_v_prime_on = True
w_prime_sq_on = True
v_prime_sq_on = True

# TKE Budget terms
tke_budget_on = True
Re_stress_component = 'uu11' # 'total' or 'uu11', 'uu12' etc. for individual components
average_z_direction = True # Averaging valid for periodic directions
average_x_direction = False
tke_production_on = True
tke_dissipation_on = True
tke_convection_on = True
tke_viscous_diffusion_on = True
tke_pressure_transport_on = True
tke_turbulent_diffusion_on = True

# Processing options ----------------------------------------------------------------------------------------------------------------------------------

# normalisation (1D data)
norm_by_u_tau_sq = True
norm_ux_by_u_tau = True
norm_y_to_y_plus = False
norm_temp_by_ref_temp = False

# Plotting options ------------------------------------------------------------------------------------------------------------------------------------

half_channel_plot = False
linear_y_scale = True
log_y_scale = False
multi_plot = True
display_fig = False
save_fig = True
save_to_path = True
plot_name = '' # name for saved plot files, leave blank for default naming

# reference data options
ux_velocity_log_ref_on = True
mhd_NK_ref_on = False # MHD turbulent channel at Re_tau=150, Ha=(4,6), Noguchi & Kasagi 1994 (thtlabs.jp)
mkm180_ch_ref_on = False # Turbulent channel at Re_tau=180, Moser, Kim & Mansour 1999 (DOI: 10.1017/S002211209900708X)

#====================================================================================================================================================
