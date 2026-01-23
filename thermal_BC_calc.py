#!/usr/bin/env python3
import numpy as np
import pandas as pd

# ====================================================================================================================================================

"""
Thermophysical properties of mediums in CHAPSim2 are provided in classes.
Functions for generating property tables or temp. difference from Grahsof number are provided.
"""

class LiquidLithiumProperties:
    """
    Thermophysical properties of liquid lithium.
    Valid range: Tm (453.65 K) to approximately 1500 K
    
    Note: Pressure effects are neglected for most properties as liquids
    are assumed incompressible. Properties are primarily temperature-dependent.
    No functions are given for properties irrelevant to CHAPSim2.

    References:
    - R.W. Ohse (Ed.) Handbook of Thermodynamic and Transport Properties of Alkali Metals, 
    Intern. Union of Pure and Applied Chemistry Chemical Data Series No. 30. Oxford: Blackwell Scientific Publ., 1985, pp. 987. 
    (All properties except Cp)
    - C.B. Alcock, M.W. Chase, V.P. Itkin, J. Phys. Chem. Ref. Data 23 (1994) 385. (Cp)
    """
    
    def __init__(self):
        self.T_melt = 453.65  # K, melting point
        self.T_boil = 1615.0  # K, boiling point at 1 atm
        self.M = 6.9410  # g/mol, molar mass of Li
        
    def phase(self, T, P):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"
    
    def density_mass(self, T):
        """Mass density in kg/m³"""
        return 278.5 - 0.04657 * T + 274.6 * (1 - T / 3500)**0.467
    
    def density_molar(self, T):
        """Molar density in mol/L"""
        rho_mass = self.density_mass(T)  # kg/m³
        rho_mol = rho_mass / self.M  # mol/m³
        return rho_mol / 1000  # mol/L
    
    def molar_volume(self, T):
        """Molar volume in L/mol"""
        return 1.0 / self.density_molar(T)
    
    def internal_energy(self, T, T_ref):
        """
        Molar internal energy in kJ/mol relative to reference temperature.
        For liquids: dU ≈ Cv*dT
        """
        Cv = self.heat_capacity_v(T)  # J/(mol·K)
        U = Cv * (T - T_ref) / 1000  # kJ/mol
        return U
    
    def enthalpy(self, T, T_ref):
        """
        delta H = int(Cp dT) from T_ref to T
        """

        return (4754 * (T - T_ref) - (0.925 * (T**2 - T_ref**2)) / 2 + (0.000291 * (T**3 - T_ref**3)) / 3) / 1000  # kJ/mol
    
    def entropy(self, T, T_ref):
        """
        Molar entropy in J/(mol·K) relative to reference temperature.
        dS = Cp/T * dT for constant pressure
        """
        Cp = self.heat_capacity_p(T)  # J/(mol·K)
        if T > T_ref:
            S = Cp * np.log(T / T_ref)
        else:
            S = 0
        return S
    
    def heat_capacity_p(self, T):
        """Molar heat capacity at constant pressure Cp in J/(mol·K)"""
        Cp_mass = 4754 - 0.925 * T + 0.000291 * T**2  # J/(kg·K)
        return Cp_mass
    
    def heat_capacity_p_molar(self, T):
        """Molar heat capacity at constant pressure Cp in J/(mol·K)"""
        Cp_mass = 4754 - 0.925 * T + 0.000291 * T**2  # J/(kg·K)
        Cp_molar = Cp_mass * self.M / 1000  # J/(mol·K)
        return Cp_molar
    
    def heat_capacity_v(self, T):
        """
        Molar heat capacity at constant volume Cv in J/(mol·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98  # Approximate
    
    def speed_of_sound(self, T):
        """
        Speed of sound in m/s
        Estimated from bulk modulus and density
        """
        # Bulk modulus for Li ~ 11-12 GPa
        K = 11.5e9  # Pa
        rho = self.density_mass(T)  # kg/m³
        c = np.sqrt(K / rho)
        return c
    
    def joule_thomson(self, T):
        """
        Joule-Thomson coefficient in K/MPa
        For liquids, typically very small and can be negative or positive
        Approximated as nearly zero
        """
        return 0.0  # Negligible for liquids
    
    def viscosity(self, T):
        """Dynamic viscosity in µPa·s"""
        mu_Pa_s = np.exp(-4.164 - 0.6374 * np.log(T) + (292.1 / T))  # Pa·s
        return mu_Pa_s * 1e6  # Convert to µPa·s
    
    def thermal_conductivity(self, T):
        """Thermal conductivity in W/(m·K)"""
        return 22.28 + 0.05 * T - 0.00001243 * T**2  # W/(m·K)
    
    def coeff_vol_exp(self, T):
        """Coefficient of volume expansion in 1/K"""
        return 1 / ( 5620 - T )

class LiquidPbLiProperties:
    """
    Thermophysical properties of liquid PbLi (Pb-17Li eutectic alloy).
    Valid range: Tm (508.0 K) to approximately 1943 K

    Note: Pressure effects are neglected for most properties as liquids
    are assumed incompressible. Properties are primarily temperature-dependent.
    No functions are given for properties irrelevant to CHAPSim2.

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 508.0   # K, melting point
        self.T_boil = 1943.0  # K, boiling point at 1 atm
        self.H_melt = 33.9e3  # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_PbLi = (10520.4, -1.1905)
        """
        return 10520.4 - 1.1905 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_PbLi = (9.148, 1.963E-2, 0.0)
        """
        return 9.148 + 1.963e-2 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_PbLi = 8836.8
        """
        return 1.0 / (8836.8 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_PbLi = (0.0, 0.0, 195.0, -9.116E-3, 0.0)
        """
        return 195.0 - 9.116e-3 * T

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98  # Approximate

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        H = HM0 + CoH(0) + CoH(1) * (T - TM0) + CoH(2) * (T^2 - TM0^2)
        CoH_PbLi = (0.0, 0.0, 195.0, -4.558E-3, 0.0)

        Integrated from Cp: H = int(Cp dT) = 195*T - 4.558E-3 * T^2 / 2
        """
        return 195.0 * (T - T_ref) - 4.558e-3 * (T**2 - T_ref**2) / 2

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) + CoM(1) * T + CoM(2) * T^2 + CoM(3) * T^3
        CoM_PbLi = (0.0061091, -2.2574E-5, 3.766E-8, -2.2887E-11)
        """
        return 0.0061091 - 2.2574e-5 * T + 3.766e-8 * T**2 - 2.2887e-11 * T**3

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidSodiumProperties:
    """
    Thermophysical properties of liquid sodium (Na).
    Valid range: Tm (371.0 K) to approximately 1155 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 371.0   # K, melting point
        self.T_boil = 1155.0  # K, boiling point at 1 atm
        self.H_melt = 113.0e3 # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_Na = (1014.0, -0.235)
        """
        return 1014.0 - 0.235 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_Na = (104.0, -0.047, 0.0)
        """
        return 104.0 - 0.047 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_Na = 4316.0
        """
        return 1.0 / (4316.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_Na = (-3.001e6, 0.0, 1658.0, -0.8479, 4.454E-4)
        """
        return -3.001e6 * T**(-2) + 1658.0 - 0.8479 * T + 4.454e-4 * T**2

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return 3.001e6 / t + 1658.0 * t - 0.8479 * t**2 / 2 + 4.454e-4 * t**3 / 3
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = exp(CoM(-1) / T + CoM(0) + CoM(1) * ln(T))
        CoM_Na = (556.835, -6.4406, -0.3958)
        """
        return np.exp(556.835 / T - 6.4406 - 0.3958 * np.log(T))

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidLeadProperties:
    """
    Thermophysical properties of liquid lead (Pb).
    Valid range: Tm (600.6 K) to approximately 2021 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 600.6   # K, melting point
        self.T_boil = 2021.0  # K, boiling point at 1 atm
        self.H_melt = 23.07e3 # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_Pb = (11441.0, -1.2795)
        """
        return 11441.0 - 1.2795 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_Pb = (9.2, 0.011, 0.0)
        """
        return 9.2 + 0.011 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_Pb = 8942.0
        """
        return 1.0 / (8942.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_Pb = (-1.524e6, 0.0, 176.2, -4.923E-2, 1.544E-5)
        """
        return -1.524e6 * T**(-2) + 176.2 - 4.923e-2 * T + 1.544e-5 * T**2

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return 1.524e6 / t + 176.2 * t - 4.923e-2 * t**2 / 2 + 1.544e-5 * t**3 / 3
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_Pb = (1069.0, 4.55E-4, 0.0)
        """
        return 4.55e-4 * np.exp(1069.0 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidBismuthProperties:
    """
    Thermophysical properties of liquid bismuth (Bi).
    Valid range: Tm (544.6 K) to approximately 1831 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 544.6   # K, melting point
        self.T_boil = 1831.0  # K, boiling point at 1 atm
        self.H_melt = 53.3e3  # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_Bi = (10725.0, -1.22)
        """
        return 10725.0 - 1.22 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_Bi = (7.34, 9.5E-3, 0.0)
        """
        return 7.34 + 9.5e-3 * T

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_BI = 8791.0
        """
        return 1.0 / (8791.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_Bi = (7.183e6, 0.0, 118.2, 5.934E-3, 0.0)
        """
        return 7.183e6 * T**(-2) + 118.2 + 5.934e-3 * T

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return -7.183e6 / t + 118.2 * t + 5.934e-3 * t**2 / 2
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_Bi = (780.0, 4.456E-4, 0.0)
        """
        return 4.456e-4 * np.exp(780.0 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidLBEProperties:
    """
    Thermophysical properties of liquid Lead-Bismuth Eutectic (LBE).
    Valid range: Tm (398.0 K) to approximately 1927 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 398.0   # K, melting point
        self.T_boil = 1927.0  # K, boiling point at 1 atm
        self.H_melt = 38.6e3  # J/kg, latent heat of melting

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_LBE = (11065.0, 1.293)
        Note: Positive coefficient - check source data
        """
        return 11065.0 + 1.293 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_LBE = (3.284, 1.617E-2, -2.305E-6)
        """
        return 3.284 + 1.617e-2 * T - 2.305e-6 * T**2

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_LBE = 8558.0
        """
        return 1.0 / (8558.0 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_LBE = (-4.56e5, 0.0, 164.8, -3.94E-2, 1.25E-5)
        """
        return -4.56e5 * T**(-2) + 164.8 - 3.94e-2 * T + 1.25e-5 * T**2

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp
        """
        def H(t):
            return 4.56e5 / t + 164.8 * t - 3.94e-2 * t**2 / 2 + 1.25e-5 * t**3 / 3
        return H(T) - H(T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_LBE = (754.1, 4.94E-4, 0.0)
        """
        return 4.94e-4 * np.exp(754.1 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6

class LiquidFLiBeProperties:
    """
    Thermophysical properties of liquid FLiBe (2LiF-BeF2 molten salt).
    Valid range: Tm (732.1 K) to approximately 1703 K

    References:
    - Correlations from CHAPSim2/src/modules.f90
    """

    def __init__(self):
        self.T_melt = 732.1   # K, melting point
        self.T_boil = 1703.0  # K, boiling point at 1 atm
        self.H_melt = 17.47e5 # J/kg, integral(Cp(TM0))

    def phase(self, T, P=None):
        """Determine phase based on temperature"""
        if T < self.T_melt:
            return "Solid"
        elif T < self.T_boil:
            return "Liquid"
        else:
            return "Vapor"

    def density_mass(self, T):
        """
        Mass density in kg/m³
        D = CoD(0) + CoD(1) * T
        CoD_FLiBe = (2413.03, -0.4884)
        """
        return 2413.03 - 0.4884 * T

    def thermal_conductivity(self, T):
        """
        Thermal conductivity in W/(m·K)
        K = CoK(0) + CoK(1) * T + CoK(2) * T^2
        CoK_FLiBe = (1.1, 0.0, 0.0)
        """
        return 1.1

    def coeff_vol_exp(self, T):
        """
        Coefficient of volume expansion in 1/K
        B = 1 / (CoB - T)
        CoB_FLiBe = 4940.7
        """
        return 1.0 / (4940.7 - T)

    def heat_capacity_p(self, T):
        """
        Specific heat capacity at constant pressure Cp in J/(kg·K)
        Cp = CoCp(-2) * T^(-2) + CoCp(-1) * T^(-1) + CoCp(0) + CoCp(1) * T + CoCp(2) * T^2
        CoCp_FLiBe = (0.0, 0.0, 2386.0, 0.0, 0.0)
        """
        return 2386.0

    def heat_capacity_v(self, T):
        """
        Specific heat capacity at constant volume Cv in J/(kg·K)
        For liquids: Cv ≈ Cp (difference is small)
        """
        return self.heat_capacity_p(T) * 0.98

    def enthalpy(self, T, T_ref):
        """
        Specific enthalpy in J/kg relative to reference temperature.
        Integrated from Cp (constant Cp)
        """
        return 2386.0 * (T - T_ref)

    def viscosity(self, T):
        """
        Dynamic viscosity in Pa·s
        M = CoM(0) * exp(CoM(-1) / T)
        CoM_FLiBe = (4022.0, 7.803E-5, 0.0)
        """
        return 7.803e-5 * np.exp(4022.0 / T)

    def viscosity_uPa_s(self, T):
        """Dynamic viscosity in µPa·s"""
        return self.viscosity(T) * 1e6


def generate_property_table(T_min, T_max, T_ref, pressure=0.1, n_points=20,
                           save_tsv=False, filename='lithium_properties.tsv'):
    """
    Generate a property table for liquid lithium over a temperature range.
    
    Parameters:
    -----------
    T_min : float
        Minimum temperature in K
    T_max : float
        Maximum temperature in K
    T_ref : float
        Reference temperature for enthalpy and entropy calculations
    pressure : float
        Pressure in MPa (default 0.1 MPa = ~1 atm)
    n_points : int
        Number of temperature points
    save_tsv : bool
        Whether to save the table as CSV
    filename : str
        Output filename if save_tsv=True
    
    Returns:
    --------
    pandas.DataFrame
        Property table with requested columns
    """
    
    li = LiquidLithiumProperties()
    
    # Check validity
    if T_min < li.T_melt:
        print(f"Warning: T_min ({T_min} K) is below melting point ({li.T_melt} K)")
    if T_max > li.T_boil:
        print(f"Warning: T_max ({T_max} K) is above boiling point ({li.T_boil} K)")
    
    # Generate temperature array
    T_array = np.linspace(T_min, T_max, n_points)
    
    # Calculate properties in the requested order
    data = {
        'Temperature (K)': T_array,
        'Pressure (MPa)': [pressure] * n_points,
        'Density (mol/l)': [li.density_molar(T) for T in T_array],
        'Volume (l/mol)': [li.molar_volume(T) for T in T_array],
        'Internal Energy (kJ/mol)': [li.internal_energy(T, T_ref) for T in T_array],
        'Enthalpy (kJ/mol)': [li.enthalpy(T, T_ref) for T in T_array],
        'Entropy (J/mol*K)': [li.entropy(T, T_ref) for T in T_array],
        'Cv (J/mol*K)': [li.heat_capacity_v(T) for T in T_array],
        'Cp (J/mol*K)': [li.heat_capacity_p_molar(T) for T in T_array],
        'Sound Spd. (m/s)': [li.speed_of_sound(T) for T in T_array],
        'Joule-Thomson (K/MPa)': [li.joule_thomson(T) for T in T_array],
        'Viscosity (uPa*s)': [li.viscosity(T) for T in T_array],
        'Therm. Cond. (W/m*K)': [li.thermal_conductivity(T) for T in T_array],
        'Phase': [li.phase(T, pressure) for T in T_array]
    }
    
    df = pd.DataFrame(data)
    
    # Save if requested
    if save_tsv:
        df.to_csv(filename, sep='\t', index=False)
        print(f"Property table saved to {filename}")
    
    return df

def Grahsof_to_temp_diff(grahsof, beta, L, mu, rho):
    """
    Convert Grahsof number to temperature difference in K.
    
    Grahsof = g * beta * Delta_T * L^3 / nu^2
    Delta_T = Grahsof * nu^2 / (g * beta * L^3)
    
    Assumptions:
    - g = 9.81 m/s²
    
    Parameters:
    -----------
    Grahsof : float
        Grahsof number
    Returns:
    """
    g = 9.81
    nu = mu / rho
    delta_T = (grahsof * nu**2) / (g * beta * L**3)
    
    return delta_T

def get_heat_flux(delta_T, k, L_ref):
    """
    Calculate heat flux from temperature difference.

    Parameters:
    -----------
    delta_T : float
        Temperature difference in K
    k : float
        Thermal conductivity in W/(m·K)
    L_ref : float
        Reference length in m

    Returns:
    --------
    float
        Heat flux in W/m²
    """
    heat_flux = k * delta_T / L_ref
    return heat_flux

def calc_fourier_number(k, rho, cp, L, timestep, nondim_time=True):
    """
    Calculate Fourier number.

    For dimensional time:    Fo = alpha * t / L^2
    For non-dimensional time (U_ref=1): Fo = alpha * t* / L
        where t_actual = t* * L (since t* = t / L with U_ref = 1 m/s)

    Parameters:
    -----------
    k : float
        Thermal conductivity in W/(m·K)
    rho : float
        Density in kg/m³
    cp : float
        Specific heat capacity at constant pressure in J/(kg·K)
    L : float
        Characteristic length in m
    timestep : float
        Time (dimensional in s, or non-dimensional if nondim_time=True)
    nondim_time : bool
        If True, timestep is non-dimensional (t* = t/L with U_ref=1 m/s)
        If False, timestep is dimensional in seconds

    Returns:
    --------
    float
        Fourier number (dimensionless)
    """
    alpha = k / (rho * cp)
    if nondim_time:
        # t_actual = t* * L, so Fo = alpha * t* * L / L^2 = alpha * t* / L
        Fo = alpha * timestep / L
    else:
        # Dimensional time: Fo = alpha * t / L^2
        Fo = alpha * timestep / (L**2)
    return Fo

def get_prandtl(temp, fluid):
    """
    Calculate Prandtl number.

    Pr = (c_p * mu) / k

    Parameters:
    -----------
    temp : float
        Temperature in K
    fluid : object
        Fluid properties object (LiquidLithiumProperties or LiquidPbLiProperties)

    Returns:
    --------
    float
        Prandtl number (dimensionless)
    """
    if isinstance(fluid, LiquidLithiumProperties):
        mu = fluid.viscosity(temp) * 1e-6  # Convert from µPa·s to Pa·s
    else:
        mu = fluid.viscosity(temp)  # Already in Pa·s
    k = fluid.thermal_conductivity(temp)
    c_p = fluid.heat_capacity_p(temp)
    Pr = (c_p * mu) / k

    return Pr

def get_fluid_properties(medium):
    """
    Get the appropriate fluid properties object based on medium name.

    Parameters:
    -----------
    medium : str
        Medium name (Li, Na, Pb, Bi, LBE, FLiBe, PbLi)

    Returns:
    --------
    object
        Fluid properties object
    """
    medium_lower = medium.lower()
    if medium_lower in ['li', 'lithium']:
        return LiquidLithiumProperties()
    elif medium_lower in ['na', 'sodium']:
        return LiquidSodiumProperties()
    elif medium_lower in ['pb', 'lead']:
        return LiquidLeadProperties()
    elif medium_lower in ['bi', 'bismuth']:
        return LiquidBismuthProperties()
    elif medium_lower in ['lbe', 'pb-bi', 'pbbi']:
        return LiquidLBEProperties()
    elif medium_lower in ['flibe', 'fli-be', '2lif-bef2']:
        return LiquidFLiBeProperties()
    elif medium_lower in ['pbli', 'pb-li', 'pbli17', 'pb17li']:
        return LiquidPbLiProperties()
    else:
        raise ValueError(f"Unknown medium: {medium}. Available options: Li, Na, Pb, Bi, LBE, FLiBe, PbLi")

def get_viscosity_Pa_s(fluid, T):
    """
    Get viscosity in Pa·s for any fluid type.
    """
    if isinstance(fluid, LiquidLithiumProperties):
        return fluid.viscosity(T) * 1e-6  # Convert from µPa·s to Pa·s
    else:
        return fluid.viscosity(T)  # Already in Pa·s

FLUID_NAMES = {
    LiquidLithiumProperties: "Liquid Lithium (Li)",
    LiquidSodiumProperties: "Liquid Sodium (Na)",
    LiquidLeadProperties: "Liquid Lead (Pb)",
    LiquidBismuthProperties: "Liquid Bismuth (Bi)",
    LiquidLBEProperties: "Liquid Lead-Bismuth Eutectic (LBE)",
    LiquidFLiBeProperties: "Liquid FLiBe (2LiF-BeF2)",
    LiquidPbLiProperties: "Liquid Lead-Lithium Eutectic (Pb-83Li-17)",
}


def interactive_calculation():
    """
    Interactive input for calculating temperature difference and heat flux
    from Grashof number, with material property summary.
    """
    print("\n" + "=" * 70)
    print("  Thermophysical Properties Calculator")
    print("=" * 70)

    # Get medium selection
    print("\nAvailable media:")
    print("  Li    - Liquid Lithium")
    print("  Na    - Liquid Sodium")
    print("  Pb    - Liquid Lead")
    print("  Bi    - Liquid Bismuth")
    print("  LBE   - Lead-Bismuth Eutectic")
    print("  FLiBe - Molten Salt (2LiF-BeF2)")
    print("  PbLi  - Lead-Lithium (Pb-17Li)")

    while True:
        medium_input = input("\nSelect medium (Li/Na/Pb/Bi/LBE/FLiBe/PbLi): ").strip()
        try:
            fluid = get_fluid_properties(medium_input)
            break
        except ValueError as e:
            print(f"Error: {e}")

    # Get Grashof number
    while True:
        try:
            grashof_input = input("Enter Grashof number (e.g., 5e7): ").strip()
            grashof = float(grashof_input)
            if grashof <= 0:
                print("Grashof number must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get reference temperature
    while True:
        try:
            T_ref = float(input(f"Enter reference temperature in K (valid range: {fluid.T_melt:.1f} - {fluid.T_boil:.1f}): ").strip())
            if T_ref < fluid.T_melt:
                print(f"Warning: Temperature below melting point ({fluid.T_melt} K)")
            if T_ref > fluid.T_boil:
                print(f"Warning: Temperature above boiling point ({fluid.T_boil} K)")
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get reference length
    while True:
        try:
            L_ref = float(input("Enter reference length in m: ").strip())
            if L_ref <= 0:
                print("Reference length must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Get timestep for Fourier number calculation
    while True:
        try:
            timestep = float(input("Enter timestep (non-dimensional, t* = t/L with U_ref=1): ").strip())
            if timestep <= 0:
                print("Timestep must be positive.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    # Calculate properties
    mu = get_viscosity_Pa_s(fluid, T_ref)
    rho = fluid.density_mass(T_ref)
    k = fluid.thermal_conductivity(T_ref)
    cp = fluid.heat_capacity_p(T_ref)
    beta = fluid.coeff_vol_exp(T_ref)
    Pr = get_prandtl(T_ref, fluid)

    # Calculate temperature difference and heat flux
    delta_T = Grahsof_to_temp_diff(grashof, beta, L_ref, mu, rho)
    heat_flux = k * delta_T / L_ref
    T_hot = T_ref + delta_T

    # Calculate Fourier number (using non-dimensional time from CHAPSim2)
    Fo = calc_fourier_number(k, rho, cp, L_ref, timestep, nondim_time=True)
    alpha = k / (rho * cp)  # thermal diffusivity for display

    # Print results
    medium_name = FLUID_NAMES.get(type(fluid), "Unknown Fluid")

    print("\n" + "=" * 70)
    print(f"  Results for {medium_name}")
    print("=" * 70)

    print("\n--- Input Parameters ---")
    print(f"  Grashof number:            {grashof:.3e}")
    print(f"  Reference temperature:     {T_ref:.2f} K")
    print(f"  Reference length:          {L_ref:.4f} m")
    print(f"  Timestep (non-dim):        {timestep:.6e}")

    print("\n--- Material Properties at T_ref ---")
    print(f"  Density (rho):             {rho:.2f} kg/m³")
    print(f"  Dynamic viscosity (mu):    {mu:.6e} Pa·s")
    print(f"  Thermal conductivity (k):  {k:.4f} W/(m·K)")
    print(f"  Specific heat (Cp):        {cp:.2f} J/(kg·K)")
    print(f"  Thermal diffusivity (α):   {alpha:.6e} m²/s")
    print(f"  Vol. expansion (beta):     {beta:.6e} 1/K")
    print(f"  Prandtl number (Pr):       {Pr:.6f}")

    print("\n--- Calculated Results ---")
    print(f"  Temperature difference:    {delta_T:.6f} K")
    print(f"  Hot wall temperature:      {T_hot:.2f} K")
    print(f"  Heat flux (q''):           {heat_flux:.4f} W/m²")
    #print(f"  Fourier number (Fo):       {Fo:.6e}")

    print("\n" + "=" * 70 + "\n")

    return {
        'grashof': grashof,
        'T_ref': T_ref,
        'L_ref': L_ref,
        'timestep': timestep,
        'delta_T': delta_T,
        'heat_flux': heat_flux,
        'Fo': Fo,
        'alpha': alpha,
        'Pr': Pr,
        'rho': rho,
        'mu': mu,
        'k': k,
        'cp': cp,
        'beta': beta
    }

if __name__ == '__main__':
    interactive_calculation()