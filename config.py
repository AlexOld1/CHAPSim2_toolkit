# Configuration file for turb_stats script ============================================================================================================

# Define input cases ==================================================================================================================================

# format: folder_path/case/1_data/quantity_timestep.dat
#folder_path = '/home/alex/sim_results/mhd_channel_validation/CPG/'
folder_path = '/home/alex/sim_results/elev_modes/'
input_format = 'visu' # 'text' (.dat) or 'visu' (.xdmf)
cases = ['isothermal_base_cases'] # case names must match folder names exactly
timesteps = ['100000']

thermo_on = False
mhd_on = False

forcing = 'CMF' # 'CMF' or 'CPG'
Re = [5000] # indexing matches 'cases' if different Re used for different cases. Use bulk reference value for CPG.
ref_temp = [670] # Kelvin

# Output ==============================================================================================================================================

# Profiles
ux_velocity_on = True
temp_on = True
tke_on = True

# Reynolds stresses
u_prime_sq_on = True
u_prime_v_prime_on = True
w_prime_sq_on = True
v_prime_sq_on = True

# TKE Budget terms
tke_budget_on = False
tke_production_on = False
tke_dissipation_on = False
tke_convection_on = False
tke_viscous_diffusion_on = False
tke_pressure_transport_on = False
tke_turbulent_diffusion_on = False

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
