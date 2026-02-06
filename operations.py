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

# Functions for computing gradient terms from t_avg data

def _extract_profile_cols(arr_2d):
    """Extract y and value columns from a [index, y, value] array."""
    if arr_2d is None:
        return None, None
    return arr_2d[:, 1], arr_2d[:, 2]

def compute_velocity_gradients(data_dict):
    """
    Compute 1D velocity gradients (∂U/∂y) from t_avg mean velocity profiles.
    
    For fully developed channel flow:
    - Flow is statistically homogeneous in x (streamwise) and z (spanwise): ∂⟨⟩/∂x = ∂⟨⟩/∂z = 0
    - Mean velocities: U₁(y), U₂ = 0, U₃ = 0
    - Therefore: ∂U₁/∂y is the only non-zero mean velocity gradient
    - All cross-stream gradients (∂U₂/∂y, ∂U₃/∂y) are computed but typically ~0 in fully developed flow

    Args:
        data_dict: Dictionary with 'u1', 'u2', 'u3' (from t_avg data, [index, y, value] format)

    Returns:
        Dictionary with 'du1dy', 'du2dy', 'du3dy' in [index, y, value] format
    """
    result = {}

    for comp in ['u1', 'u2', 'u3']:
        arr = data_dict.get(comp)
        if arr is not None:
            y, u = _extract_profile_cols(arr)
            dudy = np.gradient(u, y)
            # Return in [index, y, value] format
            result[f'd{comp}dy'] = np.column_stack([arr[:, 0], y, dudy])

    return result

def compute_tke_gradients(data_dict):
    """
    Compute TKE and its gradients from t_avg Reynolds stress profiles.
    
    For channel flow (homogeneous in x and z):
    - ∂k/∂x = 0, ∂k/∂z = 0 (homogeneity)
    - Only ∂k/∂y and ∂²k/∂y² are non-zero

    Args:
        data_dict: Dictionary with 'u1', 'u2', 'u3', 'uu11', 'uu22', 'uu33'

    Returns:
        Dictionary with 'dkdy', 'd2kdy2' in [index, y, value] format
    """
    y, u1 = _extract_profile_cols(data_dict.get('u1'))
    _, u2 = _extract_profile_cols(data_dict.get('u2'))
    _, u3 = _extract_profile_cols(data_dict.get('u3'))
    _, uu11 = _extract_profile_cols(data_dict.get('uu11'))
    _, uu22 = _extract_profile_cols(data_dict.get('uu22'))
    _, uu33 = _extract_profile_cols(data_dict.get('uu33'))

    if any(x is None for x in [y, u1, u2, u3, uu11, uu22, uu33]):
        raise ValueError("Missing required fields for TKE gradient computation")

    # Compute TKE
    k = 0.5 * ((uu11 - u1**2) + (uu22 - u2**2) + (uu33 - u3**2))
    dkdy = np.gradient(k, y)
    d2kdy2 = np.gradient(dkdy, y)

    # Return in [index, y, value] format
    index = data_dict['u1'][:, 0]
    return {
        'dkdy': np.column_stack([index, y, dkdy]),
        'd2kdy2': np.column_stack([index, y, d2kdy2]),
    }

def compute_pressure_gradient(data_dict):
    """
    Compute pressure-velocity gradient ∂⟨p'u'₂⟩/∂y from t_avg data.

    Args:
        data_dict: Dictionary with 'pr', 'u2', 'pru2'

    Returns:
        Dictionary with 'd_pu2p_dy' in [index, y, value] format, or empty if data missing
    """
    arr_u2 = data_dict.get('u2')
    arr_pr = data_dict.get('pr')
    arr_pru2 = data_dict.get('pru2')

    if arr_u2 is None or arr_pr is None or arr_pru2 is None:
        return {}

    y, u2 = _extract_profile_cols(arr_u2)
    _, pr = _extract_profile_cols(arr_pr)
    _, pru2 = _extract_profile_cols(arr_pru2)

    # Compute pressure-velocity fluctuation correlation
    pu2p = pru2 - pr * u2
    d_pu2p_dy = np.gradient(pu2p, y)

    index = arr_u2[:, 0]
    return {'d_pu2p_dy': np.column_stack([index, y, d_pu2p_dy])}

def compute_triple_correlation_gradient(data_dict):
    """
    Compute gradient of triple correlation ∂⟨u'ᵢu'ᵢu'₂⟩/∂y from t_avg data.

    Args:
        data_dict: Dictionary with 'u1', 'u2', 'u3', 'uu11', 'uu12', 'uu22', 'uu23', 'uu33'
                   and 'uuu112', 'uuu222', 'uuu233'

    Returns:
        Dictionary with 'd_uiuiu2_dy' in [index, y, value] format, or empty if data missing
    """
    arr_u1 = data_dict.get('u1')
    arr_u2 = data_dict.get('u2')
    arr_u3 = data_dict.get('u3')
    arr_uu11 = data_dict.get('uu11')
    arr_uu12 = data_dict.get('uu12')
    arr_uu22 = data_dict.get('uu22')
    arr_uu23 = data_dict.get('uu23')
    arr_uu33 = data_dict.get('uu33')
    arr_uuu112 = data_dict.get('uuu112')
    arr_uuu222 = data_dict.get('uuu222')
    arr_uuu233 = data_dict.get('uuu233')

    if any(x is None for x in [arr_u1, arr_u2, arr_u3, arr_uu11, arr_uu12, arr_uu22, arr_uu23, arr_uu33,
                                arr_uuu112, arr_uuu222, arr_uuu233]):
        return {}

    y, u1 = _extract_profile_cols(arr_u1)
    _, u2 = _extract_profile_cols(arr_u2)
    _, u3 = _extract_profile_cols(arr_u3)
    _, uu11 = _extract_profile_cols(arr_uu11)
    _, uu12 = _extract_profile_cols(arr_uu12)
    _, uu22 = _extract_profile_cols(arr_uu22)
    _, uu23 = _extract_profile_cols(arr_uu23)
    _, uu33 = _extract_profile_cols(arr_uu33)
    _, uuu112 = _extract_profile_cols(arr_uuu112)
    _, uuu222 = _extract_profile_cols(arr_uuu222)
    _, uuu233 = _extract_profile_cols(arr_uuu233)

    # Compute fluctuating triple correlations ⟨u'ᵢu'ᵢu'₂⟩
    uuu112_prime = uuu112 - uu11 * u2 - 2 * uu12 * u1 + 2 * (u1**2) * u2
    uuu222_prime = uuu222 - 3 * uu22 * u2 + 2 * (u2**3)
    uuu233_prime = uuu233 - uu33 * u2 - 2 * uu23 * u3 + 2 * (u3**2) * u2

    # Sum to get ⟨u'ᵢu'ᵢu'₂⟩
    uiuiu2p = uuu112_prime + uuu222_prime + uuu233_prime
    d_uiuiu2p_dy = np.gradient(uiuiu2p, y)

    index = arr_u1[:, 0]
    return {'d_uiuiu2_dy': np.column_stack([index, y, d_uiuiu2p_dy])}

# TKE Budget terms functions
def compute_TKE_components(xdmf_data_dict):
    """
    Compute TKE budget term components from XDMF time-space averaged data.
    Naming convention: 1,2,3 are x,y,z. 'p' denotes fluctuating component (pr used for pressure).

    Args:
        xdmf_data_dict: Dictionary containing XDMF data

    Returns:
        dict: Dictionary containing all TKE budget components
    """
    # Helper function to safely extract XDMF variables
    def get_var(name_pattern, data_type='tsp_avg'):
        """Extract variable, trying different prefixes"""
        if data_type == 'tsp_avg':
            for prefix in ['tsp_avg_flow', 'tsp_avg_thermo',
                        'tsp_avg_mhd']:
                key = f"{prefix}{name_pattern}"
                if key in xdmf_data_dict:
                    return xdmf_data_dict[key]
        if data_type == 't_avg':
            for prefix in ['t_avg_flow', 't_avg_thermo', 't_avg_mhd']:
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

    # Compute mean velocity gradients from 3D t_avg data (z,y,x)
    u1_3d = get_var('u1', data_type='t_avg')
    u2_3d = get_var('u2', data_type='t_avg')
    u3_3d = get_var('u3', data_type='t_avg')
    du_dx11 = np.gradient(u1_3d, axis = 2) if u1_3d is not None else None
    du_dx12 = np.gradient(u1_3d, axis = 1) if u1_3d is not None else None
    du_dx13 = np.gradient(u1_3d, axis = 0) if u1_3d is not None else None
    du_dx21 = np.gradient(u2_3d, axis = 2) if u2_3d is not None else None
    du_dx22 = np.gradient(u2_3d, axis = 1) if u2_3d is not None else None
    du_dx23 = np.gradient(u2_3d, axis = 0) if u2_3d is not None else None
    du_dx31 = np.gradient(u3_3d, axis = 2) if u3_3d is not None else None
    du_dx32 = np.gradient(u3_3d, axis = 1) if u3_3d is not None else None
    du_dx33 = np.gradient(u3_3d, axis = 0) if u3_3d is not None else None
    mean_velocity_grad_tensor = np.array([[du_dx11, du_dx12, du_dx13],
                                         [du_dx21, du_dx22, du_dx23],
                                         [du_dx31, du_dx32, du_dx33]])

    # Extract velocity correlations ⟨uᵢuⱼ⟩
    uu11, uu12, uu13 = get_var('uu11'), get_var('uu12'), get_var('uu13')
    uu22, uu23, uu33 = get_var('uu22'), get_var('uu23'), get_var('uu33')

    # Compute Reynolds stress fluctuations ⟨u'ᵢu'ⱼ⟩ = ⟨uᵢuⱼ⟩ - ⟨uᵢ⟩⟨uⱼ⟩ in 1D
    uu11_prime = uu11 - u1**2
    uu22_prime = uu22 - u2**2
    uu33_prime = uu33 - u3**2
    uu12_prime = uu12 - u1 * u2
    uu13_prime = uu13 - u1 * u3
    uu23_prime = uu23 - u2 * u3
    reynolds_stress_tensor = np.array([[uu11_prime, uu12_prime, uu13_prime],
                                      [uu12_prime, uu22_prime, uu23_prime],
                                      [uu13_prime, uu23_prime, uu33_prime]])

    uu11_3d, uu12_3d, uu13_3d = get_var('uu11', data_type='t_avg'), get_var('uu12', data_type='t_avg'), get_var('uu13', data_type='t_avg')
    uu22_3d, uu23_3d, uu33_3d = get_var('uu22', data_type='t_avg'), get_var('uu23', data_type='t_avg'), get_var('uu33', data_type='t_avg')

    uu11_prime_3d = uu11_3d - u1_3d**2 if uu11_3d else None
    uu22_prime_3d = uu22_3d - u2_3d**2 if uu22_3d else None
    uu33_prime_3d = uu33_3d - u3_3d**2 if uu33_3d else None
    uu12_prime_3d = uu12_3d - u1_3d * u2_3d if uu12_3d else None
    uu13_prime_3d = uu13_3d - u1_3d * u3_3d if uu13_3d else None
    uu23_prime_3d = uu23_3d - u2_3d * u3_3d if uu23_3d else None

    # Mean Convection terms: U_k ∂⟨u'_iu'_j⟩/∂x_k
    U1duu11_prime_dx1 = u1 * np.gradient(uu11_prime_3d, axis = 2) if uu11_prime_3d is not None else None
    U2duu11_prime_dx2 = u2 * np.gradient(uu11_prime_3d, axis = 1) if uu11_prime_3d is not None else None
    U3duu11_prime_dx3 = u3 * np.gradient(uu11_prime_3d, axis = 0) if uu11_prime_3d is not None else None
    
    U1duu22_prime_dx1 = u1 * np.gradient(uu22_prime_3d, axis = 2) if uu22_prime_3d is not None else None
    U2duu22_prime_dx2 = u2 * np.gradient(uu22_prime_3d, axis = 1) if uu22_prime_3d is not None else None
    U3duu22_prime_dx3 = u3 * np.gradient(uu22_prime_3d, axis = 0) if uu22_prime_3d is not None else None
    
    U1duu33_prime_dx1 = u1 * np.gradient(uu33_prime_3d, axis = 2) if uu33_prime_3d is not None else None
    U2duu33_prime_dx2 = u2 * np.gradient(uu33_prime_3d, axis = 1) if uu33_prime_3d is not None else None
    U3duu33_prime_dx3 = u3 * np.gradient(uu33_prime_3d, axis = 0) if uu33_prime_3d is not None else None
    
    U1duu12_prime_dx1 = u1 * np.gradient(uu12_prime_3d, axis = 2) if uu12_prime_3d is not None else None
    U2duu12_prime_dx2 = u2 * np.gradient(uu12_prime_3d, axis = 1) if uu12_prime_3d is not None else None
    U3duu12_prime_dx3 = u3 * np.gradient(uu12_prime_3d, axis = 0) if uu12_prime_3d is not None else None
    
    U1duu13_prime_dx1 = u1 * np.gradient(uu13_prime_3d, axis = 2) if uu13_prime_3d is not None else None
    U2duu13_prime_dx2 = u2 * np.gradient(uu13_prime_3d, axis = 1) if uu13_prime_3d is not None else None
    U3duu13_prime_dx3 = u3 * np.gradient(uu13_prime_3d, axis = 0) if uu13_prime_3d is not None else None
    
    U1duu23_prime_dx1 = u1 * np.gradient(uu23_prime_3d, axis = 2) if uu23_prime_3d is not None else None
    U2duu23_prime_dx2 = u2 * np.gradient(uu23_prime_3d, axis = 1) if uu23_prime_3d is not None else None
    U3duu23_prime_dx3 = u3 * np.gradient(uu23_prime_3d, axis = 0) if uu23_prime_3d is not None else None

    mean_conv_tensor_x1 = np.array([[U1duu11_prime_dx1, U1duu12_prime_dx1, U1duu13_prime_dx1],
                                    [U1duu12_prime_dx1, U1duu22_prime_dx1, U1duu23_prime_dx1],
                                    [U1duu13_prime_dx1, U1duu23_prime_dx1, U1duu33_prime_dx1]])
    mean_conv_tensor_x2 = np.array([[U2duu11_prime_dx2, U2duu12_prime_dx2, U2duu13_prime_dx2],
                                    [U2duu12_prime_dx2, U2duu22_prime_dx2, U2duu23_prime_dx2],
                                    [U2duu13_prime_dx2, U2duu23_prime_dx2, U2duu33_prime_dx2]])
    mean_conv_tensor_x3 = np.array([[U3duu11_prime_dx3, U3duu12_prime_dx3, U3duu13_prime_dx3],
                                    [U3duu12_prime_dx3, U3duu22_prime_dx3, U3duu23_prime_dx3],
                                    [U3duu13_prime_dx3, U3duu23_prime_dx3, U3duu33_prime_dx3]])

    # Extract pressure-velocity correlations ⟨puᵢ⟩
    pru1, pru2, pru3 = get_var('pru1'), get_var('pru2'), get_var('pru3')

    # Extract triple correlations ⟨uᵢuⱼuₖ⟩

    uuu111_3d, uuu112_3d, uuu113_3d = get_var('uuu111', data_type='t_avg'), get_var('uuu112', data_type='t_avg'), get_var('uuu113', data_type='t_avg')
    uuu122_3d, uuu123_3d, uuu133_3d = get_var('uuu122', data_type='t_avg'), get_var('uuu123', data_type='t_avg'), get_var('uuu133', data_type='t_avg')
    uuu222_3d, uuu223_3d, uuu233_3d = get_var('uuu222', data_type='t_avg'), get_var('uuu223', data_type='t_avg'), get_var('uuu233', data_type='t_avg')
    uuu333_3d = get_var('uuu333', data_type='t_avg')

    # Extract dissipation correlations ⟨∂uᵢ/∂xⱼ ∂uᵢ/∂xⱼ⟩
    dudu11, dudu12, dudu13 = get_var('dudu11'), get_var('dudu12'), get_var('dudu13')
    dudu22, dudu23, dudu33 = get_var('dudu22'), get_var('dudu23'), get_var('dudu33')

    dudu11_mean = np.sum(np.square(du_dx11), np.square(du_dx12), np.square(du_dx13))
    dudu22_mean = np.sum(np.square(du_dx21), np.square(du_dx22), np.square(du_dx23))
    dudu33_mean = np.sum(np.square(du_dx31), np.square(du_dx32), np.square(du_dx33))
    dudu12_mean = np.sum(np.multiply(du_dx11, du_dx21), np.multiply(du_dx12, du_dx22), np.multiply(du_dx13, du_dx23))
    dudu13_mean = np.sum(np.multiply(du_dx11, du_dx31), np.multiply(du_dx12, du_dx32), np.multiply(du_dx13, du_dx33))
    dudu23_mean = np.sum(np.multiply(du_dx21, du_dx31), np.multiply(du_dx22, du_dx32), np.multiply(du_dx23, du_dx33))

    dudu11_prime = dudu11 - dudu11_mean
    dudu22_prime = dudu22 - dudu22_mean
    dudu33_prime = dudu33 - dudu33_mean
    dudu12_prime = dudu12 - dudu12_mean
    dudu13_prime = dudu13 - dudu13_mean
    dudu23_prime = dudu23 - dudu23_mean

    dissipation_tensor = np.array([[dudu11_prime, dudu12_prime, dudu13_prime],
                                   [dudu12_prime, dudu22_prime, dudu23_prime],
                                   [dudu13_prime, dudu23_prime, dudu33_prime]])

    # Compute TKE = (1/2)⟨u'ᵢu'ᵢ⟩
    tke = 0.5 * (uu11_prime + uu22_prime + uu33_prime)

    # Compute pressure-velocity fluctuation correlations ⟨p'u'ᵢ⟩ = ⟨puᵢ⟩ - ⟨p⟩⟨uᵢ⟩
    pru1_prime = pru1 - pr * u1 if pru1 is not None else None
    pru2_prime = pru2 - pr * u2 if pru2 is not None else None
    pru3_prime = pru3 - pr * u3 if pru3 is not None else None

    # Compute triple correlations ⟨u'ᵢu'ⱼu'ₖ⟩ for turbulent diffusion
    # Using Reynolds decomposition: ⟨u'ᵢu'ⱼu'ₖ⟩ = ⟨uᵢuⱼuₖ⟩ - ⟨uᵢuⱼ⟩⟨uₖ⟩ - ⟨uᵢuₖ⟩⟨uⱼ⟩ - ⟨uⱼuₖ⟩⟨uᵢ⟩ + 2⟨uᵢ⟩⟨uⱼ⟩⟨uₖ⟩

    uuu111_prime_3d = uuu111_3d - 3*uu11_3d*u1_3d + 2*u1_3d**3 if uuu111_3d is not None else None
    uuu122_prime_3d = uuu122_3d - uu22_3d*u1_3d - 2*uu12_3d*u2_3d + 2*u2_3d**2*u1_3d if uuu122_3d is not None else None
    uuu123_prime_3d = uuu123_3d - uu23_3d*u1_3d - uu13_3d*u2_3d - uu12_3d*u3_3d + 2*u1_3d*u2_3d*u3_3d if uuu123_3d is not None else None
    uuu133_prime_3d = uuu133_3d - uu33_3d*u1_3d - 2*uu13_3d*u3_3d + 2*u3_3d**2*u1_3d if uuu133_3d is not None else None
    uuu112_prime_3d = uuu112_3d - uu11_3d*u2_3d - 2*uu12_3d*u1_3d + 2*u1_3d**2*u2_3d if uuu112_3d is not None else None
    uuu222_prime_3d = uuu222_3d - 3*uu22_3d*u2_3d + 2*u2_3d**3 if uuu222_3d is not None else None
    uuu233_prime_3d = uuu233_3d - uu33_3d*u2_3d - 2*uu23_3d*u3_3d + 2*u3_3d**2*u2_3d if uuu233_3d is not None else None
    uuu113_prime_3d = uuu113_3d - uu11_3d*u3_3d - 2*uu13_3d*u1_3d + 2*u1_3d**2*u3_3d if uuu113_3d is not None else None
    uuu223_prime_3d = uuu223_3d - uu22_3d*u3_3d - 2*uu23_3d*u2_3d + 2*u2_3d**2*u3_3d if uuu223_3d is not None else None
    uuu333_prime_3d = uuu333_3d - 3*uu33_3d*u3_3d + 2*u3_3d**3 if uuu333_3d is not None else None

    duuu111_prime_dx1 = np.gradient(uuu111_prime_3d, axis = 2) if uuu111_prime_3d is not None else None
    duuu121_prime_dx1 = np.gradient(uuu112_prime_3d, axis = 2) if uuu112_prime_3d is not None else None
    duuu131_prime_dx1 = np.gradient(uuu113_prime_3d, axis = 2) if uuu113_prime_3d is not None else None
    duuu221_prime_dx1 = np.gradient(uuu122_prime_3d, axis = 2) if uuu122_prime_3d is not None else None
    duuu231_prime_dx1 = np.gradient(uuu123_prime_3d, axis = 2) if uuu123_prime_3d is not None else None
    duuu331_prime_dx1 = np.gradient(uuu133_prime_3d, axis = 2) if uuu133_prime_3d is not None else None

    duuu112_prime_dx2 = np.gradient(uuu112_prime_3d, axis = 1) if uuu112_prime_3d is not None else None
    duuu122_prime_dx2 = np.gradient(uuu122_prime_3d, axis = 1) if uuu122_prime_3d is not None else None
    duuu132_prime_dx2 = np.gradient(uuu123_prime_3d, axis = 1) if uuu123_prime_3d is not None else None
    duuu222_prime_dx2 = np.gradient(uuu222_prime_3d, axis = 1) if uuu222_prime_3d is not None else None
    duuu232_prime_dx2 = np.gradient(uuu233_prime_3d, axis = 1) if uuu233_prime_3d is not None else None
    duuu332_prime_dx2 = np.gradient(uuu223_prime_3d, axis = 1) if uuu223_prime_3d is not None else None

    duuu113_prime_dx3 = np.gradient(uuu113_prime_3d, axis = 0) if uuu113_prime_3d is not None else None
    duuu123_prime_dx3 = np.gradient(uuu123_prime_3d, axis = 0) if uuu123_prime_3d is not None else None
    duuu133_prime_dx3 = np.gradient(uuu133_prime_3d, axis = 0) if uuu133_prime_3d is not None else None
    duuu233_prime_dx3 = np.gradient(uuu233_prime_3d, axis = 0) if uuu233_prime_3d is not None else None
    duuu223_prime_dx3 = np.gradient(uuu223_prime_3d, axis = 0) if uuu223_prime_3d is not None else None
    duuu333_prime_dx3 = np.gradient(uuu333_prime_3d, axis = 0) if uuu333_prime_3d is not None else None

    # turbulent diffusion tensors
    turb_conv_tensor_x1 = np.array([duuu111_prime_dx1, duuu121_prime_dx1, duuu131_prime_dx1,
                                    duuu121_prime_dx1, duuu221_prime_dx1, duuu231_prime_dx1,
                                    duuu131_prime_dx1, duuu231_prime_dx1, duuu331_prime_dx1])
    turb_conv_tensor_x2 = np.array([duuu112_prime_dx2, duuu122_prime_dx2, duuu132_prime_dx2,
                                    duuu122_prime_dx2, duuu222_prime_dx2, duuu232_prime_dx2,
                                    duuu132_prime_dx2, duuu232_prime_dx2, duuu332_prime_dx2])
    turb_conv_tensor_x3 = np.array([duuu113_prime_dx3, duuu123_prime_dx3, duuu133_prime_dx3,
                                    duuu123_prime_dx3, duuu223_prime_dx3, duuu233_prime_dx3,
                                    duuu133_prime_dx3, duuu233_prime_dx3, duuu333_prime_dx3])

    output_dict = {
        # Mean velocities
        'U1': u1, 'U2': u2, 'U3': u3,
        'pr': pr,

        # Mean velocity gradients
        'mean_velocity_grad_tensor': mean_velocity_grad_tensor,

        # Reynolds stresses (fluctuating components)
        'reynolds_stress_tensor': reynolds_stress_tensor,

        # TKE
        'TKE': tke,

        # Pressure-velocity fluctuation correlations ⟨p'u'ᵢ⟩
        'prpu1p': pru1_prime,
        'prpu2p': pru2_prime,
        'prpu3p': pru3_prime,

        # Triple velocity correlation gradient tensors for turbulent convection
        'turb_conv_tensor_x1': turb_conv_tensor_x1,
        'turb_conv_tensor_x2': turb_conv_tensor_x2,
        'turb_conv_tensor_x3': turb_conv_tensor_x3,

        # Dissipation correlations ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩
        'dissipation_tensor': dissipation_tensor,
    
        # Mean convection tensors U_k ∂⟨u'ᵢu'ⱼ⟩/∂x_k
        'mean_conv_tensor_x1': mean_conv_tensor_x1,
        'mean_conv_tensor_x2': mean_conv_tensor_x2,
        'mean_conv_tensor_x3': mean_conv_tensor_x3,
    }

    return output_dict

# tensors

def _parse_component(uiuj_str):
    """Parse 'uu12' -> (i=0, j=1)"""
    mapping = {'uu11': (0,0), 'uu12': (0,1), 'uu13': (0,2),
               'uu22': (1,1), 'uu23': (1,2), 'uu33': (2,2)}
    return mapping[uiuj_str]

def compute_production(tke_comp_dict, uiuj='total'):
    """
    Compute production term for Reynolds stress transport equation.
    
    P_ij = -<u'_i u'_k> ∂U_j/∂x_k - <u'_j u'_k> ∂U_i/∂x_k
    
    Args:
        tke_comp_dict: Dictionary from compute_TKE_components
        uiuj: Reynolds stress component ('uu11', 'uu12', 'uu22', etc.) or 'total'
    
    Returns:
        dict: Production term(s) for the specified component
    """
    
    R_ij = tke_comp_dict['reynolds_stress_tensor']
    dUi_dxj = tke_comp_dict['mean_velocity_grad_tensor']
    
    if uiuj == 'total':
        production = -np.einsum('...ij,...ij->...', R_ij, dUi_dxj) # Total TKE production computed over Re stress tensor
        return {'production': production}
    else:
        # Production for specific Reynolds stress component
        i, j = _parse_component(uiuj)
        
        # P_ij = -<u'_i u'_k> ∂U_j/∂x_k - <u'_j u'_k> ∂U_i/∂x_k
        term1 = -np.einsum('...k,...k->...', R_ij[..., i, :], dUi_dxj[..., j, :])
        term2 = -np.einsum('...k,...k->...', R_ij[..., j, :], dUi_dxj[..., i, :])
        
        production = term1 + term2
        return {f'production_{uiuj}': production}

def compute_dissipation(Re, tke_comp_dict, uiuj='total'):
    """
    Compute dissipation term for TKE budget.

    Formula: ε = -(1/Re) * ⟨∂u'ᵢ/∂xⱼ ∂u'ᵢ/∂xⱼ⟩

    Args:
        Re: Reynolds number
        tke_comp_dict: Dictionary from compute_TKE_components containing dissipation_tensor components
        uiuj: Reynolds stress component ('uu11', 'uu12', 'uu22', etc.) or 'total' for total dissipation

    Returns:
        dict: Dissipation term
    """
    
    dissipation_tensor = tke_comp_dict['dissipation_tensor']
    if uiuj == 'total':
        dissipation = -(2.0 / Re) * np.trace(dissipation_tensor)
    else:
        i, j = _parse_component(uiuj)
        dissipation = -(2.0 / Re) * dissipation_tensor[i, j]  # Dissipation for specific component

    return {'dissipation': dissipation}

def compute_mean_convection(tke_comp_dict, uiuj='total'):
    """
    Compute mean convection term
    Formula: -D/Dt<uiuj> = -Uₖ∂<uiuj>/∂xₖ, assuming no change in time.

    Args:
        tke_comp_dict: Dictionary from compute_TKE_components
        uiuj: Reynolds stress component ('uu11', 'uu12', 'uu22', etc.) or 'total'

    Returns:
        float/array: Mean Convection term
    """
    mean_conv_tensor_x1 = tke_comp_dict['mean_conv_tensor_x1']
    mean_conv_tensor_x2 = tke_comp_dict['mean_conv_tensor_x2']
    mean_conv_tensor_x3 = tke_comp_dict['mean_conv_tensor_x3']

    if uiuj == 'total':
        mean_convection = -(mean_conv_tensor_x1 + mean_conv_tensor_x2 + mean_conv_tensor_x3).trace()
    else:
        i, j = _parse_component(uiuj)
        mean_convection = -(mean_conv_tensor_x1[i, j] + mean_conv_tensor_x2[i, j] + mean_conv_tensor_x3[i, j])

    return {'mean_convection': mean_convection}

def compute_turbulent_convection(tke_comp_dict, uiuj='total'):

    turb_conv_tensor_x1 = tke_comp_dict['turb_conv_tensor_x1']
    turb_conv_tensor_x2 = tke_comp_dict['turb_conv_tensor_x2']
    turb_conv_tensor_x3 = tke_comp_dict['turb_conv_tensor_x3']

    if uiuj == 'total':
        turb_convection = -(turb_conv_tensor_x1 + turb_conv_tensor_x2 + turb_conv_tensor_x3).trace()
    else:        
        i, j = _parse_component(uiuj)
        turb_convection = -(turb_conv_tensor_x1[i, j] + turb_conv_tensor_x2[i, j] + turb_conv_tensor_x3[i, j])

    return {'turbulent_convection': turb_convection}

def compute_viscous_diffusion(Re):
    """
    Compute viscous diffusion term for TKE budget.

    Formula: VD = 

    Args:
        Re: Reynolds number

    Returns:
        float/array: Viscous diffusion term
    """
    visc_diff = None

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
    
    # Requires 

    #press_strain = 

    return {'pressure_transport': press_transport}


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