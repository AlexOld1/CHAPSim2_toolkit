import numpy as np

# Reynolds number functions
def get_Re(case, cases, Re, ux_velocity, flow_forcing):
    if flow_forcing == 'CMF':
        if not int and len(Re) > 1:
            cur_Re = Re[cases.index(case)]
        else:
         cur_Re = Re[0]
    elif flow_forcing == 'CPG':
        if not int and len(Re) > 1:
            cur_Re = Re[cases.index(case)] * (0.5 * np.trapezoid(ux_velocity[:, 2], ux_velocity[:, 1]))
        else:
           cur_Re = Re[0] * (0.5 * np.trapezoid(ux_velocity[:, 2], ux_velocity[:, 1]))
    else:
        raise ValueError("flow_forcing must be either 'CMF' or 'CPG'")
    return cur_Re

def get_ref_Re(case, cases, Re):
    if not int and len(Re) > 1:
        ref_Re = Re[cases.index(case)]
    else:
        ref_Re = Re[0]
    return ref_Re

# Profiles, Reynolds stresses and total TKE functions
def read_profile(ux):
    ux = ux[:, 2]
    return ux

def compute_normal_stress(ux, uu):
    ux_col = ux[:, 2]
    uu_col = uu[:, 2]

    return uu_col - np.square(ux_col)

def compute_shear_stress(ux, uy, uv):
    ux_col = ux[:, 2]
    uv_col = uv[:, 2]
    uy_col = uy[:, 2]

    ux_uy = np.multiply(ux_col, uy_col)
    return uv_col - ux_uy

def compute_tke(u_prime_sq, v_prime_sq, w_prime_sq):
    return 0.5 * (u_prime_sq + v_prime_sq + w_prime_sq)

# TKE Budget terms functions
def compute_TKE_components(xdmf_data_dict):
    """
    Compute TKE budget term components from XDMF time-space averaged data.

    This function extracts all available data from XDMF files and computes:
    - Reynolds stresses
    - Pressure-velocity correlations
    - Triple velocity correlations
    - Dissipation terms
    - TKE and derived quantities

    Args:
        xdmf_data_dict: Dictionary containing XDMF data with keys like 'tsp_avg_flow_cell_u1', etc.

    Returns:
        dict: Dictionary containing all TKE budget components
    """
    # Helper function to safely extract XDMF variables
    def get_var(name_pattern):
        """Extract variable, trying different prefixes"""
        for prefix in ['tsp_avg_flow_cell_tsp_avg_', 'time_avg_cell_tsp_avg_', 'flow_cell_tsp_avg_']:
            key = f"{prefix}{name_pattern}"
            if key in xdmf_data_dict:
                return xdmf_data_dict[key]
        # Fallback: try without prefix
        if name_pattern in xdmf_data_dict:
            return xdmf_data_dict[name_pattern]
        return None

    # Extract mean velocities and pressure
    u1, u2, u3 = get_var('u1'), get_var('u2'), get_var('u3')
    pr = get_var('pr')

    # Extract Reynolds stresses ⟨uᵢuⱼ⟩
    uu11, uu12, uu13 = get_var('uu11'), get_var('uu12'), get_var('uu13')
    uu22, uu23, uu33 = get_var('uu22'), get_var('uu23'), get_var('uu33')

    # Extract pressure-velocity correlations ⟨p'u'ᵢ⟩
    pru1, pru2, pru3 = get_var('pru1'), get_var('pru2'), get_var('pru3')

    # Extract triple correlations ⟨uᵢuⱼuₖ⟩
    uuu111 = get_var('uuu111')
    uuu112, uuu113 = get_var('uuu112'), get_var('uuu113')
    uuu122, uuu123, uuu133 = get_var('uuu122'), get_var('uuu123'), get_var('uuu133')
    uuu222, uuu223, uuu233 = get_var('uuu222'), get_var('uuu223'), get_var('uuu233')
    uuu333 = get_var('uuu333')

    # Extract dissipation correlations ⟨∂uᵢ/∂xⱼ ∂uᵢ/∂xⱼ⟩
    dudu11, dudu12, dudu13 = get_var('dudu11'), get_var('dudu12'), get_var('dudu13')
    dudu22, dudu23, dudu33 = get_var('dudu22'), get_var('dudu23'), get_var('dudu33')

    # Compute Reynolds stress fluctuations ⟨u'ᵢu'ⱼ⟩ = ⟨uᵢuⱼ⟩ - ⟨uᵢ⟩⟨uⱼ⟩
    u1p_u1p = uu11 - u1**2
    u2p_u2p = uu22 - u2**2
    u3p_u3p = uu33 - u3**2
    u1p_u2p = uu12 - u1 * u2
    u1p_u3p = uu13 - u1 * u3
    u2p_u3p = uu23 - u2 * u3

    # Compute TKE = (1/2)⟨u'ᵢu'ᵢ⟩
    tke = 0.5 * (u1p_u1p + u2p_u2p + u3p_u3p)

    # Compute pressure-velocity fluctuation correlations ⟨p'u'ᵢ⟩ = ⟨puᵢ⟩ - ⟨p⟩⟨uᵢ⟩
    pu1p = pru1 - pr * u1 if pru1 is not None else None
    pu2p = pru2 - pr * u2 if pru2 is not None else None
    pu3p = pru3 - pr * u3 if pru3 is not None else None

    # Compute triple correlations ⟨u'ᵢu'ⱼu'ₖ⟩ for turbulent diffusion
    # Using Reynolds decomposition: ⟨u'ᵢu'ⱼu'ₖ⟩ = ⟨uᵢuⱼuₖ⟩ - ⟨uᵢuⱼ⟩⟨uₖ⟩ - ⟨uᵢuₖ⟩⟨uⱼ⟩ - ⟨uⱼuₖ⟩⟨uᵢ⟩ + 2⟨uᵢ⟩⟨uⱼ⟩⟨uₖ⟩

    # For turbulent diffusion term: ⟨u'ᵢu'ᵢu'ⱼ⟩ components
    # ⟨u'₁u'₁u'₁⟩ = ⟨u₁u₁u₁⟩ - 3⟨u₁u₁⟩⟨u₁⟩ + 2⟨u₁⟩³
    u1pu1pu1p = uuu111 - 3*uu11*u1 + 2*u1**3 if uuu111 is not None else None

    # ⟨u'₁u'₁u'₂⟩ = ⟨u₁u₁u₂⟩ - ⟨u₁u₁⟩⟨u₂⟩ - 2⟨u₁u₂⟩⟨u₁⟩ + 2⟨u₁⟩²⟨u₂⟩
    u1pu1pu2p = uuu112 - uu11*u2 - 2*uu12*u1 + 2*u1**2*u2 if uuu112 is not None else None

    # ⟨u'₁u'₁u'₃⟩ = ⟨u₁u₁u₃⟩ - ⟨u₁u₁⟩⟨u₃⟩ - 2⟨u₁u₃⟩⟨u₁⟩ + 2⟨u₁⟩²⟨u₃⟩
    u1pu1pu3p = uuu113 - uu11*u3 - 2*uu13*u1 + 2*u1**2*u3 if uuu113 is not None else None

    # ⟨u'₂u'₂u'₁⟩ = ⟨u₂u₂u₁⟩ - ⟨u₂u₂⟩⟨u₁⟩ - 2⟨u₂u₁⟩⟨u₂⟩ + 2⟨u₂⟩²⟨u₁⟩
    u2pu2pu1p = uuu122 - uu22*u1 - 2*uu12*u2 + 2*u2**2*u1 if uuu122 is not None else None

    # ⟨u'₂u'₂u'₂⟩ = ⟨u₂u₂u₂⟩ - 3⟨u₂u₂⟩⟨u₂⟩ + 2⟨u₂⟩³
    u2pu2pu2p = uuu222 - 3*uu22*u2 + 2*u2**3 if uuu222 is not None else None

    # ⟨u'₂u'₂u'₃⟩ = ⟨u₂u₂u₃⟩ - ⟨u₂u₂⟩⟨u₃⟩ - 2⟨u₂u₃⟩⟨u₂⟩ + 2⟨u₂⟩²⟨u₃⟩
    u2pu2pu3p = uuu223 - uu22*u3 - 2*uu23*u2 + 2*u2**2*u3 if uuu223 is not None else None

    # ⟨u'₃u'₃u'₁⟩ = ⟨u₃u₃u₁⟩ - ⟨u₃u₃⟩⟨u₁⟩ - 2⟨u₃u₁⟩⟨u₃⟩ + 2⟨u₃⟩²⟨u₁⟩
    u3pu3pu1p = uuu133 - uu33*u1 - 2*uu13*u3 + 2*u3**2*u1 if uuu133 is not None else None

    # ⟨u'₃u'₃u'₂⟩ = ⟨u₃u₃u₂⟩ - ⟨u₃u₃⟩⟨u₂⟩ - 2⟨u₃u₂⟩⟨u₃⟩ + 2⟨u₃⟩²⟨u₂⟩
    u3pu3pu2p = uuu233 - uu33*u2 - 2*uu23*u3 + 2*u3**2*u2 if uuu233 is not None else None

    # ⟨u'₃u'₃u'₃⟩ = ⟨u₃u₃u₃⟩ - 3⟨u₃u₃⟩⟨u₃⟩ + 2⟨u₃⟩³
    u3pu3pu3p = uuu333 - 3*uu33*u3 + 2*u3**3 if uuu333 is not None else None

    # For turbulent diffusion: ⟨u'ᵢu'ᵢu'ⱼ⟩ summed over i
    # ⟨u'ᵢu'ᵢu'₁⟩ = ⟨u'₁u'₁u'₁⟩ + ⟨u'₂u'₂u'₁⟩ + ⟨u'₃u'₃u'₁⟩
    uiuiu1p = None
    if all(x is not None for x in [u1pu1pu1p, u2pu2pu1p, u3pu3pu1p]):
        uiuiu1p = u1pu1pu1p + u2pu2pu1p + u3pu3pu1p

    # ⟨u'ᵢu'ᵢu'₂⟩ = ⟨u'₁u'₁u'₂⟩ + ⟨u'₂u'₂u'₂⟩ + ⟨u'₃u'₃u'₂⟩
    uiuiu2p = None
    if all(x is not None for x in [u1pu1pu2p, u2pu2pu2p, u3pu3pu2p]):
        uiuiu2p = u1pu1pu2p + u2pu2pu2p + u3pu3pu2p

    # ⟨u'ᵢu'ᵢu'₃⟩ = ⟨u'₁u'₁u'₃⟩ + ⟨u'₂u'₂u'₃⟩ + ⟨u'₃u'₃u'₃⟩
    uiuiu3p = None
    if all(x is not None for x in [u1pu1pu3p, u2pu2pu3p, u3pu3pu3p]):
        uiuiu3p = u1pu1pu3p + u2pu2pu3p + u3pu3pu3p

    output_dict = {
        # Mean velocities
        'U1': u1, 'U2': u2, 'U3': u3,
        'pr': pr,

        # Reynolds stresses (fluctuating components)
        'u1p_u1p': u1p_u1p,
        'u2p_u2p': u2p_u2p,
        'u3p_u3p': u3p_u3p,
        'u1p_u2p': u1p_u2p,
        'u2p_u1p': u1p_u2p,  # Symmetric
        'u1p_u3p': u1p_u3p,
        'u3p_u1p': u1p_u3p,  # Symmetric
        'u2p_u3p': u2p_u3p,
        'u3p_u2p': u2p_u3p,  # Symmetric

        # TKE
        'TKE': tke,

        # Pressure-velocity fluctuation correlations ⟨p'u'ᵢ⟩
        'pu1p': pu1p,
        'pu2p': pu2p,
        'pu3p': pu3p,

        # Individual triple correlations ⟨u'ᵢu'ᵢu'ⱼ⟩
        'u1pu1pu1p': u1pu1pu1p, 'u1pu1pu2p': u1pu1pu2p, 'u1pu1pu3p': u1pu1pu3p,
        'u2pu2pu1p': u2pu2pu1p, 'u2pu2pu2p': u2pu2pu2p, 'u2pu2pu3p': u2pu2pu3p,
        'u3pu3pu1p': u3pu3pu1p, 'u3pu3pu2p': u3pu3pu2p, 'u3pu3pu3p': u3pu3pu3p,

        # Summed triple correlations for turbulent diffusion: ⟨u'ᵢu'ᵢu'ⱼ⟩
        'uiuiu1p': uiuiu1p,
        'uiuiu2p': uiuiu2p,
        'uiuiu3p': uiuiu3p,

        # Dissipation correlations ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩
        # These are already fluctuating quantities from the XDMF file
        'dudu11': dudu11, 'dudu12': dudu12, 'dudu13': dudu13,
        'dudu22': dudu22, 'dudu23': dudu23, 'dudu33': dudu33,
    }

    return output_dict

def compute_production(tke_comp_dict):
    """
    Compute production term for TKE budget: P_i = -<u'_i u'_j> * dU_i/dx_j

    Args:
        tke_comp_dict: Dictionary from compute_TKE_components

    Returns:
        dict: Production terms for each component and total
    """
    prod_u1 = -1 * (tke_comp_dict['u1p_u1p'] * tke_comp_dict['du1dx'] +
                    tke_comp_dict['u1p_u2p'] * tke_comp_dict['du1dy'] +
                    tke_comp_dict['u1p_u3p'] * tke_comp_dict['du1dz'])

    prod_u2 = -1 * (tke_comp_dict['u2p_u1p'] * tke_comp_dict['du2dx'] +
                    tke_comp_dict['u2p_u2p'] * tke_comp_dict['du2dy'] +
                    tke_comp_dict['u2p_u3p'] * tke_comp_dict['du2dz'])

    prod_u3 = -1 * (tke_comp_dict['u3p_u1p'] * tke_comp_dict['du3dx'] +
                    tke_comp_dict['u3p_u2p'] * tke_comp_dict['du3dy'] +
                    tke_comp_dict['u3p_u3p'] * tke_comp_dict['du3dz'])

    return {
        'prod_u1': prod_u1,
        'prod_u2': prod_u2,
        'prod_u3': prod_u3,
        'prod_total': prod_u1 + prod_u2 + prod_u3
    }

def compute_dissipation(Re, tke_comp_dict):
    """
    Compute dissipation term for TKE budget.

    Formula: ε = -(1/Re) * ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩

    The XDMF data provides:
    - dudu11 = ⟨∂u₁/∂x₁ ∂u₁/∂x₁⟩ (and similar for other components)

    The total dissipation is the sum over all velocity components and all directions:
    ε = -(1/Re) * Σᵢ Σⱼ ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩

    Args:
        Re: Reynolds number
        tke_comp_dict: Dictionary from compute_TKE_components containing dudu11, dudu22, dudu33

    Returns:
        dict: Dissipation term
    """
    # Extract dissipation correlations
    dudu11 = tke_comp_dict.get('dudu11')
    dudu12 = tke_comp_dict.get('dudu12')
    dudu13 = tke_comp_dict.get('dudu13')
    dudu22 = tke_comp_dict.get('dudu22')
    dudu23 = tke_comp_dict.get('dudu23')
    dudu33 = tke_comp_dict.get('dudu33')

    # Check if all required components are available
    if any(x is None for x in [dudu11, dudu12, dudu13, dudu22, dudu23, dudu33]):
        raise ValueError("Missing dissipation correlation data (dudu terms) in XDMF file")

    # Total dissipation: sum of all gradient correlations
    # ε = -(1/Re) * [⟨(∂u₁/∂x₁)²⟩ + ⟨(∂u₁/∂x₂)²⟩ + ⟨(∂u₁/∂x₃)²⟩ +
    #                ⟨(∂u₂/∂x₁)²⟩ + ⟨(∂u₂/∂x₂)²⟩ + ⟨(∂u₂/∂x₃)²⟩ +
    #                ⟨(∂u₃/∂x₁)²⟩ + ⟨(∂u₃/∂x₂)²⟩ + ⟨(∂u₃/∂x₃)²⟩]
    # The XDMF files store:
    # dudu11 = sum over j of ⟨(∂u₁/∂xⱼ)²⟩
    # dudu22 = sum over j of ⟨(∂u₂/∂xⱼ)²⟩
    # dudu33 = sum over j of ⟨(∂u₃/∂xⱼ)²⟩
    total_dissipation = dudu11 + dudu22 + dudu33

    dissipation = -(1.0 / Re) * total_dissipation

    return {'dissipation': dissipation}

def compute_convection(mean_velocities, tke_gradients):
    """
    Compute convection (advection) term for TKE budget.

    Formula: C = -Uⱼ ∂k/∂xⱼ = -U₁∂k/∂x - U₂∂k/∂y - U₃∂k/∂z

    Args:
        mean_velocities: Dictionary with 'U1', 'U2', 'U3'
        tke_gradients: Dictionary with 'dkdx', 'dkdy', 'dkdz'

    Returns:
        float/array: Convection term
    """
    convection = -(mean_velocities['U1'] * tke_gradients['dkdx'] +
                   mean_velocities['U2'] * tke_gradients['dkdy'] +
                   mean_velocities['U3'] * tke_gradients['dkdz'])

    return {'convection': convection}

def compute_viscous_diffusion(Re, tke_second_derivs):
    """
    Compute viscous diffusion term for TKE budget.

    Formula: VD = (1/Re) * ∂²k/∂xⱼ² = (1/Re) * (∂²k/∂x² + ∂²k/∂y² + ∂²k/∂z²)

    Args:
        Re: Reynolds number
        tke_second_derivs: Dictionary with 'd2kdx2', 'd2kdy2', 'd2kdz2'

    Returns:
        float/array: Viscous diffusion term
    """
    visc_diff = (1.0 / Re) * (tke_second_derivs['d2kdx2'] +
                               tke_second_derivs['d2kdy2'] +
                               tke_second_derivs['d2kdz2'])

    return {'viscous_diffusion': visc_diff}

def compute_pressure_transport(pressure_velocity_corr_grads):
    """
    Compute pressure transport term for TKE budget.

    Formula: PT = -∂⟨p'u'ⱼ⟩/∂xⱼ = -(∂⟨p'u'₁⟩/∂x + ∂⟨p'u'₂⟩/∂y + ∂⟨p'u'₃⟩/∂z)

    Args:
        pressure_velocity_corr_grads: Dictionary with 'd_pu1p_dx', 'd_pu2p_dy', 'd_pu3p_dz'

    Returns:
        float/array: Pressure transport term
    """
    press_transport = -(pressure_velocity_corr_grads['d_pu1p_dx'] +
                        pressure_velocity_corr_grads['d_pu2p_dy'] +
                        pressure_velocity_corr_grads['d_pu3p_dz'])

    return {'pressure_transport': press_transport}

def compute_turbulent_diffusion(triple_corr_grads):
    """
    Compute turbulent diffusion term for TKE budget.

    Formula: TD = -(1/2) * ∂⟨u'ᵢu'ᵢu'ⱼ⟩/∂xⱼ
           = -(1/2) * [∂⟨u'ᵢu'ᵢu'₁⟩/∂x + ∂⟨u'ᵢu'ᵢu'₂⟩/∂y + ∂⟨u'ᵢu'ᵢu'₃⟩/∂z]

    Args:
        triple_corr_grads: Dictionary with gradients of triple correlations
                          'd_uiuiu1_dx', 'd_uiuiu2_dy', 'd_uiuiu3_dz'

    Returns:
        float/array: Turbulent diffusion term
    """
    turb_diff = -0.5 * (triple_corr_grads['d_uiuiu1_dx'] +
                        triple_corr_grads['d_uiuiu2_dy'] +
                        triple_corr_grads['d_uiuiu3_dz'])

    return {'turbulent_diffusion': turb_diff}

def compute_buoyancy_term():
    return

def compute_mhd_term():
    return

# Normalisation & Averaging functions
def norm_turb_stat_wrt_u_tau_sq(ux_data, turb_stat, Re_bulk):

    Re_bulk = int(Re_bulk)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    tau_w = dudy/Re_bulk
    u_tau_sq = abs(tau_w)
    turb_stat = np.asarray(turb_stat)
    return np.divide(turb_stat, u_tau_sq)

def norm_ux_velocity_wrt_u_tau(ux_data, Re_bulk):
    
    Re_bulk = int(Re_bulk)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    u_tau = np.sqrt(abs(dudy/Re_bulk))
    ux_velocity = ux_data[:, 2]
    ux_velocity = np.asarray(ux_velocity)
    return np.divide(ux_velocity, u_tau)

def norm_y_to_y_plus(y, ux_data, Re_bulk):

    Re_bulk = int(Re_bulk)
    du = ux_data[0, 2] - ux_data[1, 2]
    dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du/dy
    u_tau = np.sqrt(abs(dudy/Re_bulk))
    y_plus = y * u_tau * Re_bulk

    return y_plus

def symmetric_average(arr):
    n = len(arr)
    half = n // 2
    # If even length: average symmetric pairs
    if n % 2 == 0:
        return (arr[:half] + arr[::-1][:half]) / 2
    else:
        # Odd length: do same, keep middle element as is
        symmetric_avg = (arr[:half] + arr[::-1][:half]) / 2
        middle = np.array([(arr[half])])  # keep center value
        return np.concatenate((symmetric_avg, middle))

def window_average(data_t1, data_t2, t1, t2, stat_start_timestep):

    stat_t2 = t2 - stat_start_timestep
    stat_t1 = t1 - stat_start_timestep
    t_diff = stat_t2 - stat_t1

    if t_diff == 0:
        return data_t2  # If no difference, return the second timestep data directly
    else:
        return (stat_t2 * data_t2 - stat_t1 * data_t1) / t_diff

def analytical_laminar_mhd_prof(case, Re_bulk, Re_tau): # U. Müller, L. Bühler, Analytical Solutions for MHD Channel Flow, 2001.
        u_tau = Re_tau / Re_bulk
        y = np.linspace(0, 1, 100) * Re_tau
        prof = (((Re_tau * u_tau)/(case * np.tanh(case)))*((1 - np.cosh(case * (1 - y)))/np.cosh(case)) + 1.225)
        return prof

# Vorticity functions
def compute_vorticity_omega_x(uy, uz, y, z): # test vorticity functions

    duzdy = np.gradient(uz, y)
    duydz = np.gradient(uy, z)
    omega_x = duzdy - duydz

    return omega_x

def compute_vorticity_omega_y(ux, uz, x, z):

    duxdz = np.gradient(ux, z)
    duzdx = np.gradient(uz, x)
    omega_y = duxdz- duzdx

    return omega_y

def compute_vorticity_omega_z(uy, ux, x, y):

    duydx = np.gradient(uy, x)
    duxdy = np.gradient(ux, y)
    omega_z = duydx - duxdy

    return omega_z


# lamda2 and q criterion next