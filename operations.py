import numpy as np

# =====================================================================================================================================================
# Helper: extract value from native array or legacy 3-column format
# =====================================================================================================================================================

def _extract_val(data):
    """Return the value array from either native or legacy 3-column [idx, y, val] format."""
    if data.ndim == 2 and data.shape[1] == 3:
        return data[:, 2]
    return data

def _compute_u_tau_quantities(ux_data, Re_bulk, y_coords=None):
    """Compute wall shear stress quantities from near-wall velocity data."""
    Re_bulk = int(Re_bulk)
    if y_coords is not None:
        du = ux_data[0] - ux_data[1]
        dy = y_coords[0] - y_coords[1]
    else:
        du = ux_data[0, 2] - ux_data[1, 2]
        dy = ux_data[0, 1] - ux_data[1, 1]
    dudy = du / dy
    tau_w = dudy / Re_bulk
    u_tau_sq = abs(tau_w)
    u_tau = np.sqrt(u_tau_sq)
    return u_tau, u_tau_sq, tau_w

# =====================================================================================================================================================
# Reynolds number functions
# =====================================================================================================================================================

def get_Re(case, cases, Re, ux_velocity, flow_forcing, y_coords=None):
    if flow_forcing == 'CMF':
        if not int and len(Re) > 1:
            cur_Re = Re[cases.index(case)]
        else:
         cur_Re = Re[0]
    elif flow_forcing == 'CPG':
        profile = _extract_val(ux_velocity) if y_coords is None else ux_velocity
        if y_coords is not None:
            y = y_coords
        else:
            y = ux_velocity[:, 1]
        if not int and len(Re) > 1:
            cur_Re = Re[cases.index(case)] * (0.5 * np.trapezoid(profile, y))
        else:
           cur_Re = Re[0] * (0.5 * np.trapezoid(profile, y))
    else:
        raise ValueError("flow_forcing must be either 'CMF' or 'CPG'")
    return cur_Re

def get_ref_Re(case, cases, Re):
    if not int and len(Re) > 1:
        ref_Re = Re[cases.index(case)]
    else:
        ref_Re = Re[0]
    return ref_Re

# =====================================================================================================================================================
# Profiles, Reynolds stresses and total TKE functions
# =====================================================================================================================================================

def read_profile(ux):
    return _extract_val(ux)

def compute_normal_stress(ux, uu):
    ux_col = _extract_val(ux)
    uu_col = _extract_val(uu)
    return uu_col - np.square(ux_col)

def compute_shear_stress(ux, uy, uv):
    ux_col = _extract_val(ux)
    uy_col = _extract_val(uy)
    uv_col = _extract_val(uv)
    return uv_col - np.multiply(ux_col, uy_col)

def compute_tke(u_prime_sq, v_prime_sq, w_prime_sq):
    return 0.5 * (u_prime_sq + v_prime_sq + w_prime_sq)

# =====================================================================================================================================================
# TKE Budget terms functions
# =====================================================================================================================================================

def second_derivative(f, y, axis=0):
    """Second derivative on a non-uniform y mesh. Operates along *axis* (default 0)."""
    d2f = np.empty_like(f)
    h = np.diff(y)
    h1 = h[:-1]  # left spacing
    h2 = h[1:]   # right spacing

    # Build generic slicers for the requested axis
    def _sl(s):
        idx = [slice(None)] * f.ndim
        idx[axis] = s
        return tuple(idx)

    # Broadcast h1, h2 to the correct axis
    shape = [1] * f.ndim
    shape[axis] = -1
    h1b = h1.reshape(shape)
    h2b = h2.reshape(shape)

    d2f[_sl(slice(1, -1))] = (
        2 * (f[_sl(slice(2, None))] * h1b
             - f[_sl(slice(1, -1))] * (h1b + h2b)
             + f[_sl(slice(None, -2))] * h2b)
        / (h1b * h2b * (h1b + h2b))
    )
    # Boundaries: copy neighbour
    d2f[_sl(slice(0, 1))] = d2f[_sl(slice(1, 2))]
    d2f[_sl(slice(-1, None))] = d2f[_sl(slice(-2, -1))]
    return d2f


def compute_TKE_components(xdmf_data_dict, y_coords, average_z=False, average_x=False):
    """
    Compute TKE budget term components from XDMF data.
    Naming convention: 1,2,3 are x,y,z. 'prime' denotes fluctuating component.

    Supports:
        3D data (nz, ny, nx)          — average_z=False
        2D data (ny, nx)  z-averaged  — average_z=True
        1D data (ny,)     xz-averaged — average_z=True, average_x=True

    Args:
        xdmf_data_dict: Dictionary containing XDMF data (prefix-stripped names)
        y_coords: 1D array of y cell-centre coordinates
        average_z: True if z has been averaged out (data is 2D or 1D)
        average_x: True if x has been averaged out (data is 1D)

    Returns:
        dict: Dictionary containing all TKE budget tensors
    """

    # ------------------------------------------------------------------
    # Gradient helpers — axis mapping depends on data dimensionality
    # 3D (nz,ny,nx): x=axis2, y=axis1, z=axis0
    # 2D (ny,nx):    x=axis1, y=axis0, z=n/a
    # 1D (ny,):      x=n/a,   y=axis0, z=n/a
    # ------------------------------------------------------------------
    def grad_x(field):
        if field is None:
            return None
        if average_x:
            return np.zeros_like(field)
        if average_z:  # 2D (ny, nx)
            return np.gradient(field, axis=1)
        return np.gradient(field, axis=2)  # 3D (nz, ny, nx)

    def grad_y(field):
        if field is None:
            return None
        if field.ndim == 1:
            return np.gradient(field, y_coords)
        if average_z:  # 2D (ny, nx)
            return np.gradient(field, y_coords, axis=0)
        return np.gradient(field, y_coords, axis=1)  # 3D (nz, ny, nx)

    def grad_z(field):
        if field is None:
            return None
        if average_z:
            return np.zeros_like(field)
        return np.gradient(field, axis=0)  # 3D (nz, ny, nx)

    def lap_y(field):
        """Second derivative in y on a stretched mesh."""
        if field is None:
            return None
        if field.ndim == 1:
            return second_derivative(field, y_coords, axis=0)
        if average_z:  # 2D
            return second_derivative(field, y_coords, axis=0)
        return second_derivative(field, y_coords, axis=1)  # 3D

    # ------------------------------------------------------------------
    # Variable lookup (prefixes already stripped by reader)
    # ------------------------------------------------------------------
    def get_var(name):
        return xdmf_data_dict.get(name, None)

    # ------------------------------------------------------------------
    # Mean velocities and pressure
    # ------------------------------------------------------------------
    u1, u2, u3 = get_var('u1'), get_var('u2'), get_var('u3')
    pr = get_var('pr')

    # Mean velocity gradient tensor dU_i/dx_j  (shape: 3,3,...)
    du_dx = [[None]*3 for _ in range(3)]
    u_fields = [u1, u2, u3]
    grad_fns = [grad_x, grad_y, grad_z]
    for i in range(3):
        for j in range(3):
            du_dx[i][j] = grad_fns[j](u_fields[i])

    mean_velocity_grad_tensor = np.array(du_dx)

    # ------------------------------------------------------------------
    # Velocity correlations ⟨uᵢuⱼ⟩ and Reynolds stress fluctuations
    # ------------------------------------------------------------------
    uu_names = {(0,0): 'uu11', (0,1): 'uu12', (0,2): 'uu13',
                (1,1): 'uu22', (1,2): 'uu23', (2,2): 'uu33'}

    uu = {}
    for (i, j), name in uu_names.items():
        uu[i, j] = uu[j, i] = get_var(name)

    # ⟨u'ᵢu'ⱼ⟩ = ⟨uᵢuⱼ⟩ − ⟨uᵢ⟩⟨uⱼ⟩
    uu_prime = {}
    for (i, j), name in uu_names.items():
        val = uu[i, j] - u_fields[i] * u_fields[j] if uu[i, j] is not None else None
        uu_prime[i, j] = uu_prime[j, i] = val

    reynolds_stress_tensor = np.array([
        [uu_prime[0,0], uu_prime[0,1], uu_prime[0,2]],
        [uu_prime[0,1], uu_prime[1,1], uu_prime[1,2]],
        [uu_prime[0,2], uu_prime[1,2], uu_prime[2,2]],
    ])

    # ------------------------------------------------------------------
    # Fluctuating velocity rms and its gradient tensor
    # ------------------------------------------------------------------
    u_prime_rms = [None]*3
    for i in range(3):
        if uu_prime[i, i] is not None:
            u_prime_rms[i] = np.sqrt(np.clip(uu_prime[i, i], 0, None))

    fluc_grad = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            fluc_grad[i][j] = grad_fns[j](u_prime_rms[i])
    fluc_velocity_grad_tensor = np.array(fluc_grad)

    # ------------------------------------------------------------------
    # Reynolds stress gradients  d⟨u'ᵢu'ⱼ⟩/dx_k
    # ------------------------------------------------------------------
    dR = {}  # (i, j, k) -> array
    for (i, j) in uu_names.keys():
        for k in range(3):
            dR[i, j, k] = dR[j, i, k] = grad_fns[k](uu_prime[i, j])

    # Mean convection tensors: U_k * d⟨u'ᵢu'ⱼ⟩/dx_k
    def _build_sym_tensor(func):
        """Build (3,3,...) tensor from func(i,j)."""
        return np.array([[func(i, j) for j in range(3)] for i in range(3)])

    mean_conv_tensor_x1 = _build_sym_tensor(lambda i, j: u_fields[0] * dR[i, j, 0] if dR[i, j, 0] is not None else np.zeros_like(u_fields[0]))
    mean_conv_tensor_x2 = _build_sym_tensor(lambda i, j: u_fields[1] * dR[i, j, 1] if dR[i, j, 1] is not None else np.zeros_like(u_fields[1]))
    mean_conv_tensor_x3 = _build_sym_tensor(lambda i, j: u_fields[2] * dR[i, j, 2] if dR[i, j, 2] is not None else np.zeros_like(u_fields[2]))

    # ------------------------------------------------------------------
    # Laplacian of Reynolds stresses  ∂²⟨u'ᵢu'ⱼ⟩/∂x_k²
    # ------------------------------------------------------------------
    def _lap_component(i, j, k):
        """Second derivative of R_ij in direction k."""
        field = uu_prime[i, j]
        if field is None:
            return None
        if k == 0:   # x
            return grad_x(dR[i, j, 0]) if dR[i, j, 0] is not None else None
        elif k == 1: # y
            return lap_y(field)
        else:        # z
            return grad_z(dR[i, j, 2]) if dR[i, j, 2] is not None else None

    lap_re_stress_tensor_x1 = _build_sym_tensor(lambda i, j: _lap_component(i, j, 0) if _lap_component(i, j, 0) is not None else np.zeros_like(u1))
    lap_re_stress_tensor_x2 = _build_sym_tensor(lambda i, j: _lap_component(i, j, 1) if _lap_component(i, j, 1) is not None else np.zeros_like(u1))
    lap_re_stress_tensor_x3 = _build_sym_tensor(lambda i, j: _lap_component(i, j, 2) if _lap_component(i, j, 2) is not None else np.zeros_like(u1))

    # ------------------------------------------------------------------
    # Pressure-velocity fluctuation correlations
    # ------------------------------------------------------------------
    pru = [get_var('pru1'), get_var('pru2'), get_var('pru3')]
    pru_prime = [None]*3
    for i in range(3):
        if pru[i] is not None and pr is not None:
            pru_prime[i] = pru[i] - pr * u_fields[i]

    # p' estimate
    if pru_prime[0] is not None and uu_prime[0, 0] is not None:
        denom = np.sqrt(np.clip(uu_prime[0, 0], 1e-30, None))
        pr_prime = pru_prime[0] / denom
    else:
        pr_prime = np.zeros_like(u1) if u1 is not None else None

    # ∂⟨p'u'ᵢ⟩/∂x_j
    press_grad = [[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            press_grad[i][j] = grad_fns[j](pru_prime[i])
    press_velocity_fluc_grad_tensor = np.array(press_grad)

    # ------------------------------------------------------------------
    # Dissipation tensor  ⟨(∂u'ᵢ/∂xₖ)(∂u'ⱼ/∂xₖ)⟩
    # ------------------------------------------------------------------
    dudu_names = {(0,0): 'dudu11', (0,1): 'dudu12', (0,2): 'dudu13',
                  (1,1): 'dudu22', (1,2): 'dudu23', (2,2): 'dudu33'}

    dudu = {}
    for (i, j), name in dudu_names.items():
        dudu[i, j] = dudu[j, i] = get_var(name)

    # Mean part: sum_k (dUi/dxk)(dUj/dxk)
    dudu_mean = {}
    for (i, j) in dudu_names.keys():
        val = None
        for k in range(3):
            if du_dx[i][k] is not None and du_dx[j][k] is not None:
                term = du_dx[i][k] * du_dx[j][k]
                val = term if val is None else val + term
        dudu_mean[i, j] = dudu_mean[j, i] = val

    dudu_prime = {}
    for (i, j) in dudu_names.keys():
        if dudu[i, j] is not None and dudu_mean[i, j] is not None:
            dudu_prime[i, j] = dudu_prime[j, i] = dudu[i, j] - dudu_mean[i, j]
        else:
            dudu_prime[i, j] = dudu_prime[j, i] = dudu.get((i, j))

    dissipation_tensor = np.array([
        [dudu_prime.get((0,0), np.zeros_like(u1)), dudu_prime.get((0,1), np.zeros_like(u1)), dudu_prime.get((0,2), np.zeros_like(u1))],
        [dudu_prime.get((0,1), np.zeros_like(u1)), dudu_prime.get((1,1), np.zeros_like(u1)), dudu_prime.get((1,2), np.zeros_like(u1))],
        [dudu_prime.get((0,2), np.zeros_like(u1)), dudu_prime.get((1,2), np.zeros_like(u1)), dudu_prime.get((2,2), np.zeros_like(u1))],
    ])

    # ------------------------------------------------------------------
    # TKE
    # ------------------------------------------------------------------
    tke = 0.5 * (uu_prime[0,0] + uu_prime[1,1] + uu_prime[2,2]) if uu_prime[0,0] is not None else None

    # ------------------------------------------------------------------
    # Triple correlations ⟨u'ᵢu'ⱼu'ₖ⟩ and turbulent convection
    # Using: ⟨u'ᵢu'ⱼu'ₖ⟩ = ⟨uᵢuⱼuₖ⟩ − ⟨uᵢuⱼ⟩⟨uₖ⟩ − ⟨uᵢuₖ⟩⟨uⱼ⟩ − ⟨uⱼuₖ⟩⟨uᵢ⟩ + 2⟨uᵢ⟩⟨uⱼ⟩⟨uₖ⟩
    # ------------------------------------------------------------------
    uuu_raw = {
        (1,1,1): get_var('uuu111'), (1,1,2): get_var('uuu112'), (1,1,3): get_var('uuu113'),
        (1,2,2): get_var('uuu122'), (1,2,3): get_var('uuu123'), (1,3,3): get_var('uuu133'),
        (2,2,2): get_var('uuu222'), (2,2,3): get_var('uuu223'), (2,3,3): get_var('uuu233'),
        (3,3,3): get_var('uuu333'),
    }

    def _triple_prime(i1, j1, k1):
        """Compute ⟨u'_i u'_j u'_k⟩ from raw triple and lower-order correlations."""
        key = tuple(sorted([i1, j1, k1]))
        raw = uuu_raw.get(key)
        if raw is None:
            return None
        u_i, u_j, u_k = u_fields[i1-1], u_fields[j1-1], u_fields[k1-1]
        uu_ij = uu.get((i1-1, j1-1))
        uu_ik = uu.get((i1-1, k1-1))
        uu_jk = uu.get((j1-1, k1-1))
        if any(v is None for v in [u_i, u_j, u_k, uu_ij, uu_ik, uu_jk]):
            return None
        return raw - uu_ij*u_k - uu_ik*u_j - uu_jk*u_i + 2*u_i*u_j*u_k

    # Turbulent convection: −∂⟨u'ᵢu'ⱼu'ₖ⟩/∂xₖ
    # Need d/dx_k of ⟨u'_i u'_j u'_k⟩ for each (i,j) summed over k
    # Build tensors for each k-direction
    def _turb_conv_component(i, j, k):
        """d⟨u'_{i+1} u'_{j+1} u'_{k+1}⟩/dx_{k+1}"""
        tp = _triple_prime(i+1, j+1, k+1)
        return grad_fns[k](tp)

    def _zero():
        return np.zeros_like(u1) if u1 is not None else 0.0

    turb_conv_tensor_x1 = _build_sym_tensor(lambda i, j: _turb_conv_component(i, j, 0) if _turb_conv_component(i, j, 0) is not None else _zero())
    turb_conv_tensor_x2 = _build_sym_tensor(lambda i, j: _turb_conv_component(i, j, 1) if _turb_conv_component(i, j, 1) is not None else _zero())
    turb_conv_tensor_x3 = _build_sym_tensor(lambda i, j: _turb_conv_component(i, j, 2) if _turb_conv_component(i, j, 2) is not None else _zero())

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    return {
        'U1': u1, 'U2': u2, 'U3': u3,
        'pr': pr,
        'TKE': tke,
        'mean_velocity_grad_tensor': mean_velocity_grad_tensor,
        'fluc_velocity_grad_tensor': fluc_velocity_grad_tensor,
        'reynolds_stress_tensor': reynolds_stress_tensor,
        'lap_re_stress_tensor_x1': lap_re_stress_tensor_x1,
        'lap_re_stress_tensor_x2': lap_re_stress_tensor_x2,
        'lap_re_stress_tensor_x3': lap_re_stress_tensor_x3,
        'pr_prime': pr_prime,
        'press_velocity_fluc_grad_tensor': press_velocity_fluc_grad_tensor,
        'turb_conv_tensor_x1': turb_conv_tensor_x1,
        'turb_conv_tensor_x2': turb_conv_tensor_x2,
        'turb_conv_tensor_x3': turb_conv_tensor_x3,
        'dissipation_tensor': dissipation_tensor,
        'mean_conv_tensor_x1': mean_conv_tensor_x1,
        'mean_conv_tensor_x2': mean_conv_tensor_x2,
        'mean_conv_tensor_x3': mean_conv_tensor_x3,
    }


# =====================================================================================================================================================
# Budget term extraction functions (dimension-agnostic on (3,3,...) tensors)
# =====================================================================================================================================================

def _parse_component(uiuj_str):
    """Parse 'uu12' -> (i=0, j=1)"""
    mapping = {'uu11': (0,0), 'uu12': (0,1), 'uu13': (0,2),
               'uu22': (1,1), 'uu23': (1,2), 'uu33': (2,2)}
    return mapping[uiuj_str]

def compute_production(tke_comp_dict, uiuj='total'):
    """
    Production: P_ij = -R_ik dU_j/dx_k - R_jk dU_i/dx_k
    Tensors have shape (3, 3, ...) where first two axes are i,j.
    """
    R = tke_comp_dict['reynolds_stress_tensor']       # (3,3,...)
    dU = tke_comp_dict['mean_velocity_grad_tensor']    # (3,3,...)

    if uiuj == 'total':
        # P_total = -R_ik dU_i/dx_k  (contract both i and k)
        production = -np.einsum('ik...,ik...->...', R, dU)
        return {'production': production}
    else:
        i, j = _parse_component(uiuj)
        # P_ij = -sum_k [ R_{ik} dU_j/dx_k + R_{jk} dU_i/dx_k ]
        term1 = -np.einsum('k...,k...->...', R[i], dU[j])
        term2 = -np.einsum('k...,k...->...', R[j], dU[i])
        return {f'production_{uiuj}': term1 + term2}

def compute_dissipation(Re, tke_comp_dict, uiuj='total'):
    """Dissipation: ε_ij = -(2/Re) * ⟨(∂u'_i/∂x_k)(∂u'_j/∂x_k)⟩"""
    D = tke_comp_dict['dissipation_tensor']
    if uiuj == 'total':
        dissipation = -(2.0 / Re) * np.trace(D)
    else:
        i, j = _parse_component(uiuj)
        dissipation = -(2.0 / Re) * D[i, j]
    return {'dissipation': dissipation}

def compute_mean_convection(tke_comp_dict, uiuj='total'):
    """Mean convection: -U_k ∂⟨u'_i u'_j⟩/∂x_k"""
    C1 = tke_comp_dict['mean_conv_tensor_x1']
    C2 = tke_comp_dict['mean_conv_tensor_x2']
    C3 = tke_comp_dict['mean_conv_tensor_x3']
    total = C1 + C2 + C3
    if uiuj == 'total':
        return {'mean_convection': -np.trace(total)}
    else:
        i, j = _parse_component(uiuj)
        return {'mean_convection': -total[i, j]}

def compute_turbulent_convection(tke_comp_dict, uiuj='total'):
    """Turbulent convection: -∂⟨u'_i u'_j u'_k⟩/∂x_k"""
    T1 = tke_comp_dict['turb_conv_tensor_x1']
    T2 = tke_comp_dict['turb_conv_tensor_x2']
    T3 = tke_comp_dict['turb_conv_tensor_x3']
    total = T1 + T2 + T3
    if uiuj == 'total':
        return {'turbulent_convection': -np.trace(total)}
    else:
        i, j = _parse_component(uiuj)
        return {'turbulent_convection': -total[i, j]}

def compute_viscous_diffusion(Re, turb_comp_dict, uiuj='total'):
    """Viscous diffusion: (1/Re) * ∇²⟨u'_i u'_j⟩"""
    L1 = turb_comp_dict['lap_re_stress_tensor_x1']
    L2 = turb_comp_dict['lap_re_stress_tensor_x2']
    L3 = turb_comp_dict['lap_re_stress_tensor_x3']
    total = L1 + L2 + L3
    if uiuj == 'total':
        return {'viscous_diffusion': (1/Re) * np.trace(total)}
    else:
        i, j = _parse_component(uiuj)
        return {'viscous_diffusion': (1/Re) * total[i, j]}

def compute_pressure_transport(tke_comp_dict, uiuj='total'):
    """Pressure transport: -(∂⟨p'u'_j⟩/∂x_i + ∂⟨p'u'_i⟩/∂x_j)"""
    P = tke_comp_dict['press_velocity_fluc_grad_tensor']
    if uiuj == 'total':
        return {'pressure_transport': -2 * np.trace(P)}
    else:
        i, j = _parse_component(uiuj)
        return {'pressure_transport': -(P[i, j] + P[j, i])}

def compute_pressure_strain(tke_comp_dict, uiuj='total'):
    """Pressure strain: ⟨p'(∂u'_i/∂x_j + ∂u'_j/∂x_i)⟩"""
    pr_prime = tke_comp_dict['pr_prime']
    G = tke_comp_dict['fluc_velocity_grad_tensor']
    if uiuj == 'total':
        return {'pressure_strain': 0}
    else:
        i, j = _parse_component(uiuj)
        return {'pressure_strain': pr_prime * (G[i, j] + G[j, i])}

def compute_buoyancy_term():
    return

def compute_mhd_term():
    return

# =====================================================================================================================================================
# Normalisation & Averaging functions
# =====================================================================================================================================================

def norm_turb_stat_wrt_u_tau_sq(ux_data, turb_stat, Re_bulk, y_coords=None):
    _, u_tau_sq, _ = _compute_u_tau_quantities(ux_data, Re_bulk, y_coords)
    return np.divide(np.asarray(turb_stat), u_tau_sq)

def norm_ux_velocity_wrt_u_tau(ux_data, Re_bulk, y_coords=None):
    u_tau, _, _ = _compute_u_tau_quantities(ux_data, Re_bulk, y_coords)
    profile = ux_data if y_coords is not None else ux_data[:, 2]
    return np.divide(np.asarray(profile), u_tau)

def norm_y_to_y_plus(y, ux_data, Re_bulk, y_coords=None):
    Re_bulk = int(Re_bulk)
    u_tau, _, _ = _compute_u_tau_quantities(ux_data, Re_bulk, y_coords)
    return y * u_tau * Re_bulk

def symmetric_average(arr):
    n = len(arr)
    half = n // 2
    if n % 2 == 0:
        return (arr[:half] + arr[::-1][:half]) / 2
    else:
        symmetric_avg = (arr[:half] + arr[::-1][:half]) / 2
        middle = np.array([(arr[half])])
        return np.concatenate((symmetric_avg, middle))

def window_average(data_t1, data_t2, t1, t2, stat_start_timestep):
    stat_t2 = t2 - stat_start_timestep
    stat_t1 = t1 - stat_start_timestep
    t_diff = stat_t2 - stat_t1
    if t_diff == 0:
        return data_t2
    else:
        return (stat_t2 * data_t2 - stat_t1 * data_t1) / t_diff

def analytical_laminar_mhd_prof(case, Re_bulk, Re_tau):
    u_tau = Re_tau / Re_bulk
    y = np.linspace(0, 1, 100) * Re_tau
    prof = (((Re_tau * u_tau)/(case * np.tanh(case)))*((1 - np.cosh(case * (1 - y)))/np.cosh(case)) + 1.225)
    return prof

# =====================================================================================================================================================
# Vorticity functions
# =====================================================================================================================================================

def compute_vorticity_omega_x(uy, uz, y, z):
    duzdy = np.gradient(uz, y)
    duydz = np.gradient(uy, z)
    return duzdy - duydz

def compute_vorticity_omega_y(ux, uz, x, z):
    duxdz = np.gradient(ux, z)
    duzdx = np.gradient(uz, x)
    return duxdz - duzdx

def compute_vorticity_omega_z(uy, ux, x, y):
    duydx = np.gradient(uy, x)
    duxdy = np.gradient(ux, y)
    return duydx - duxdy

# lamda2 and q criterion next
