"""
Flow Assure — VLP Module (Modified Hagedorn-Brown)

All functions use **field units**:
- Flow rate  :  STB/day (oil)
- Pressure   :  psia
- Depth      :  ft
- Diameter   :  inches
- Temperature:  °F
- Viscosity  :  cP
- Density    :  lb/ft³  (lbm/ft³)
- Roughness  :  inches
- Surface tension : dynes/cm

"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from backend.pvt import (
    calculate_rs_standing,
    calculate_rs_vasquez_beggs,
    calculate_bo_standing,
    calculate_bo_vasquez_beggs,
    calculate_bo_above_pb,
    calculate_dead_oil_viscosity_beggs_robinson,
    calculate_live_oil_viscosity_beggs_robinson,
    calculate_z_factor_dak,
    calculate_bg,
    calculate_gas_viscosity_lge,
    calculate_muw,
)

from backend.beggs_brill import pressure_gradient_beggs_brill

#  CONSTANTS

BBL_TO_FT3 = 5.615          # 1 barrel = 5.615 ft³
DAY_TO_SEC = 86400.0        # seconds per day
G_C        = 32.174         # lbm·ft / (lbf·s²)
PI         = math.pi


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE PVT HELPERS  (for pressure-traverse use)
# ══════════════════════════════════════════════════════════════════════════════

def _api_to_sg(api: float) -> float:
    """Oil API gravity → specific gravity (water = 1)."""
    return 141.5 / (api + 131.5)


def _gas_density(P: float, T: float, gas_sg: float, Z: float) -> float:
    """Gas density, lb/ft³.   ρg = 28.97·γg·P / (Z·10.73·T_R)"""
    T_R = T + 459.67
    denom = Z * 10.73 * T_R
    if denom <= 0:
        return 0.01
    return 28.97 * gas_sg * P / denom


def _oil_density(api: float, Rs: float, Bo: float, gas_sg: float) -> float:
    """Oil density at in-situ conditions, lb/ft³."""
    gamma_o = _api_to_sg(api)
    return max((62.4 * gamma_o + 0.0136 * Rs * gas_sg) / max(Bo, 1.0), 1.0)


def _water_density(water_sg: float = 1.0) -> float:
    """Water density, lb/ft³ (simplified)."""
    return 62.4 * water_sg


def _surface_tension(api: float, T: float, Rs: float) -> float:
    """Baker-Swerdloff oil–gas surface tension, dynes/cm."""
    sigma_68 = max(39.0 - 0.2571 * api, 1.0)
    sigma_100 = max(37.5 - 0.2571 * api, 1.0)

    if T <= 68:
        sigma_dead = sigma_68
    elif T >= 100:
        sigma_dead = sigma_100
    else:
        sigma_dead = sigma_68 + (sigma_100 - sigma_68) * (T - 68.0) / 32.0

    correction = max(math.exp(-Rs / 1200.0), 0.05)
    return max(sigma_dead * correction, 1.0)


def _get_fluid_props(
    P: float, T: float, q_oil: float, q_water: float, GOR: float,
    d_in: float, d_ft: float, A_pipe: float, api: float, gas_sg: float,
    water_sg: float, pvt_method: str = "Standing",
    Psep: float = 100.0, Tsep: float = 75.0, salinity: float = 0.0,
    Pb: Optional[float] = None,
    apply_bo_undersat_correction: bool = False,
    Rs_manual: Optional[float] = None,
    Bo_manual: Optional[float] = None,
    mu_o_manual: Optional[float] = None,
) -> dict:
    """Generates the fluid property dictionary `fp` for the gradient calculations."""
    pb_for_rs = Pb if (
        Pb is not None and Pb > 0
    ) else None
    
    if pvt_method == "Vasquez-Beggs":
        rs_calc, _ = calculate_rs_vasquez_beggs(
            P, T, api, gas_sg,
            Psep=Psep,
            Tsep=Tsep,
            Pb=pb_for_rs,
        )
    else:
        rs_calc, _ = calculate_rs_standing(
            P, T, api, gas_sg,
            Pb=pb_for_rs,
        )

    # Manual Rs override for VLP.
    # Still cap by producing GOR so free_GOR = GOR - Rs does not become negative.
    if Rs_manual is not None and Rs_manual > 0:
        Rs = min(float(Rs_manual), GOR)
    else:
        Rs = min(rs_calc, GOR)

    # Calculate Bo.
    # Default behavior: use the existing saturated Bo correlation.
    # Optional behavior: if enabled and P > Pb, calculate Bo at Pb first,
    # then apply undersaturated Bo correction.
    if (
        apply_bo_undersat_correction
        and Pb is not None
        and Pb > 0
        and P > Pb
    ):
        if Rs_manual is not None and Rs_manual > 0:
            Rs_at_Pb = min(float(Rs_manual), GOR)
        else:
            if pvt_method == "Vasquez-Beggs":
                rs_pb_calc, _ = calculate_rs_vasquez_beggs(
                    Pb, T, api, gas_sg,
                    Psep=Psep,
                    Tsep=Tsep,
                )
            else:
                rs_pb_calc, _ = calculate_rs_standing(
                    Pb, T, api, gas_sg,
                )
    
            Rs_at_Pb = min(rs_pb_calc, GOR)
    
        if pvt_method == "Vasquez-Beggs":
            Bo_at_Pb, _ = calculate_bo_vasquez_beggs(
                Rs_at_Pb, T, api, gas_sg,
                Psep=Psep,
                Tsep=Tsep,
            )
        else:
            Bo_at_Pb, _ = calculate_bo_standing(
                Rs_at_Pb, T, api, gas_sg,
            )
    
        Bo, _ = calculate_bo_above_pb(
            Bo_at_Pb=Bo_at_Pb,
            P=P,
            Pb=Pb,
            Rs_at_Pb=Rs_at_Pb,
            T=T,
            api=api,
            gas_sg=gas_sg,
        )
    
    else:
        if pvt_method == "Vasquez-Beggs":
            Bo, _ = calculate_bo_vasquez_beggs(
                Rs, T, api, gas_sg,
                Psep=Psep,
                Tsep=Tsep,
            )
        else:
            Bo, _ = calculate_bo_standing(
                Rs, T, api, gas_sg,
            )
    
    # Manual Bo override for VLP.
    # Manual Bo has highest priority.
    if Bo_manual is not None and Bo_manual > 0:
        Bo = float(Bo_manual)

    Z, _ = calculate_z_factor_dak(P, T, gas_sg)

    mu_od, _ = calculate_dead_oil_viscosity_beggs_robinson(api, T)
    mu_o, _ = calculate_live_oil_viscosity_beggs_robinson(mu_od, Rs)

    # Manual oil viscosity override for VLP.
    if mu_o_manual is not None and mu_o_manual > 0:
        mu_o = float(mu_o_manual)

    mu_g, _ = calculate_gas_viscosity_lge(P, T, gas_sg, Z)
    mu_w, _ = calculate_muw(P, T, salinity=salinity)

    rho_o = _oil_density(api, Rs, Bo, gas_sg)
    rho_g = _gas_density(P, T, gas_sg, Z)
    rho_w = _water_density(water_sg)
    sigma = _surface_tension(api, T, Rs)

    q_oil_is  = q_oil * Bo * BBL_TO_FT3 / DAY_TO_SEC
    q_water_is = q_water * 1.0 * BBL_TO_FT3 / DAY_TO_SEC
    free_GOR = max(GOR - Rs, 0.0)
    Bg_rb, _ = calculate_bg(P, T, Z)
    Bg = Bg_rb * BBL_TO_FT3
    q_gas_is = q_oil * free_GOR * Bg / DAY_TO_SEC

    q_liq = q_oil_is + q_water_is
    q_liq = max(q_liq, 1e-15)

    v_sl = q_liq / A_pipe if A_pipe > 0 else 0.0
    v_sg = q_gas_is / A_pipe if A_pipe > 0 else 0.0
    v_m  = v_sl + v_sg

    lam_l = v_sl / v_m if v_m > 1e-10 else 1.0

    if free_GOR <= 0 or v_sg < 1e-10:
        v_sg = 0.0
        lam_l = 1.0
        v_m = v_sl

    return {
        'p': P,
        't': T,
        'd_in': d_in,
        'rho_l': max(rho_o * (q_oil_is/q_liq) + rho_w * (q_water_is/q_liq), 1.0),
        'rho_g': max(rho_g, 0.01),
        'mu_l': max(mu_o * (q_oil_is/q_liq) + mu_w * (q_water_is/q_liq), 0.01),
        'mu_g': max(mu_g, 1e-6),
        'sigma': max(sigma, 0.1),
        'v_sl': v_sl,
        'v_sg': v_sg,
        'v_m': v_m,
        'lam_l': lam_l,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION FACTOR CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _friction_churchill(Re, rough, d_in):
    """
    Churchill Friction Factor
    """
    if Re <= 0: return 0.025

    e_d = rough / d_in

    # Churchill parameters
    A = (-2.457 * math.log((7.0 / Re)**0.9 + 0.27 * e_d)) ** 16
    B = (37530.0 / Re) ** 16

    # Darcy friction factor
    f_D = 8.0 * ((8.0 / Re)**12 + 1.0 / (A + B)**1.5) ** (1.0 / 12.0)

    return f_D


def _friction_chen(Re, rough, d_in):
    """
    Chen Explicit Friction Factor
    """
    if Re <= 0:   return 0.025
    if Re < 2100: return 64.0 / Re       # Darcy friction factor, laminar
    e = rough / d_in                      # relative roughness

    # Chen's explicit formula for turbulent friction factor (Darcy)
    tmp = -2.0 * math.log10(
        e / 3.7065
        - (5.0452 / Re) * math.log10(e**1.1098 / 2.8257 + (7.149 / Re)**0.8981)
    )
    return (1.0 / tmp)**2                 # Darcy friction factor, turbulent


def _friction(Re, rough, d_in, friction_method="Churchill"):
    """Dispatcher: picks the friction factor correlation based on method string."""
    method = (friction_method or "Churchill").strip().lower()

    if "chen" in method:
        return _friction_chen(Re, rough, d_in)

    return _friction_churchill(Re, rough, d_in)


# ══════════════════════════════════════════════════════════════════════════════
#  H-B CHART POLYNOMIALS / DIGITIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _CNl(Nl):
    Nl = max(min(Nl, 10.0), 1e-6)
    return max(0.061*Nl**3 - 0.0929*Nl**2 + 0.0505*Nl + 0.0019, 1e-6)


_H_PTS = np.array([1e-7, 1e-6, 3e-6, 1e-5, 7.12e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 1.0])
_HLPSI_PTS = np.array([0.03, 0.08, 0.09, 0.20, 0.44, 0.4896, 0.65, 0.80, 0.92, 0.98, 1.00])
_LOG_H_PTS = np.log10(_H_PTS)


def _holdup_over_psi(H, use_polynomial=False):
    H = max(H, 1e-9)
    if use_polynomial:
        # User requested Empirical Rational Polynomial formula for H-B Chart 3
        # HL/psi = [ (0.0047 + 1123.32H + 729489.64H^2) / (1 + 1097.1566H + 722153.97H^2) ]^0.5
        num = 0.0047 + 1123.32 * H + 729489.64 * (H ** 2)
        den = 1.0 + 1097.1566 * H + 722153.97 * (H ** 2)
        return float((num / den) ** 0.5)
    else:
        # Default: highly accurate digitized log-log interpolation of Chart 3
        log_H = math.log10(H)
        return float(np.interp(log_H, _LOG_H_PTS, _HLPSI_PTS))


def _psi(B):
    if B <= 0.025:
        return max(27170*B**3 - 317.52*B**2 + 0.5472*B + 0.9999, 1.0)
    elif B <= 0.055:
        return max(-533.33*B**2 + 58.524*B + 0.1171, 1.0)
    else:
        return max(2.5714*B + 1.5962, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSIONLESS GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def _dim_groups(fp):
    rho_l = fp['rho_l']
    sigma = max(fp['sigma'], 0.1)
    mu_l  = fp['mu_l']
    d_ft  = fp['d_in'] / 12.0    #d is converted into feet.
    v_sl  = fp['v_sl']
    v_sg  = fp['v_sg']

    Nl  = 0.15726 * mu_l * (1.0 / (rho_l * sigma**3))**0.25
    Nlv = 1.938   * v_sl * (rho_l / sigma)**0.25
    Ngv = 1.938   * v_sg * (rho_l / sigma)**0.25
    Nd  = 120.872 * d_ft * math.sqrt(rho_l / sigma)

    return Nl, Nlv, Ngv, Nd


# ══════════════════════════════════════════════════════════════════════════════
#  LIQUID HOLDUP CALCULATORS
# ══════════════════════════════════════════════════════════════════════════════

def _griffith_holdup(fp):
    """
    Griffith Bubble Flow Liquid Holdup

    """
    # Original Griffith constant slip velocity for bubbles (ft/s)
    bubble_slip_velocity = 0.8
    
    mixture_velocity = fp['v_m']
    superficial_gas_velocity = fp['v_sg']
    no_slip_liquid_fraction = fp['lam_l']

    # Solve the quadratic equation for liquid holdup in bubble flow
    discriminant = (1.0 + mixture_velocity / bubble_slip_velocity)**2 - 4.0 * superficial_gas_velocity / bubble_slip_velocity
    discriminant = max(discriminant, 0.0)
    
    liquid_holdup = 1.0 - 0.5 * (1.0 + mixture_velocity / bubble_slip_velocity - math.sqrt(discriminant))
    
    # Physics constraint: Liquid holdup can never be less than the no-slip liquid fraction
    return max(min(liquid_holdup, 1.0), no_slip_liquid_fraction)


def _hb_holdup(fp, use_polynomial=False):
    """
    Standard Hagedorn-Brown Holdup
    """
    pressure_psia = fp['p']
    no_slip_liquid_fraction = fp['lam_l']

    # Calculate the 4 primary dimensionless groups (Viscosity, Liquid Velocity, Gas Velocity, Diameter)
    Nl, Nlv, Ngv, Nd = _dim_groups(fp)
    
    # Cap the viscosity number between 1e-6 and 10 to stay within chart limits
    Nl = max(min(Nl, 10.0), 1e-6)

    # 1. Calculate the 'H' correlating parameter (x-axis of Chart 3)
    if Ngv > 0 and Nd > 0:
        liquid_viscosity_correction = _CNl(Nl)
        H = (Nlv / Ngv**0.575) * (pressure_psia / 14.7)**0.1 * liquid_viscosity_correction / Nd 
    else:
        H = 0.0
        
    # 2. Calculate the 'B' correlating parameter (x-axis of Chart 4)
    if Nd > 0:
        B = (Ngv * Nlv**0.38) / Nd**2.14 
    else:
        B = 0.0

    # 3. Look up the holdup factor (HL/psi) from Chart 3
    holdup_factor = _holdup_over_psi(H, use_polynomial)
    
    # 4. Look up the secondary correction factor (psi) from Chart 4
    secondary_correction = _psi(B)
    
    # 5. Calculate final liquid holdup
    liquid_holdup = holdup_factor * secondary_correction
    
    # Physics constraint: Liquid holdup can never be less than the no-slip liquid fraction
    return max(min(liquid_holdup, 1.0), no_slip_liquid_fraction)


def _blended_holdup(fp, use_polynomial=False):
    """
    To transition from Griffith Bubble Flow to Hagedorn-Brown we used
    this smoothing logic.
    """
    mixture_velocity = fp['v_m']
    pipe_id_inches = fp['d_in']
    no_slip_liquid_fraction = fp['lam_l']
    no_slip_gas_fraction = 1.0 - no_slip_liquid_fraction

    # Calculate the Griffith Bubble Flow Boundary (L_B)
    bubble_boundary = 1.071 - (0.2662 * (mixture_velocity**2) / pipe_id_inches)
    bubble_boundary = max(bubble_boundary, 0.13)

    # Define the transition window: between 70% of boundary and 100% of boundary
    pure_griffith_threshold = 0.7 * bubble_boundary
    pure_hb_threshold = bubble_boundary

    # Zone 1: Pure Griffith Bubble Flow (Very low gas)
    if no_slip_gas_fraction < pure_griffith_threshold:
        return _griffith_holdup(fp), 'griffith_bubble_flow'

    # Zone 2: Pure Hagedorn-Brown (High gas - Slug/Mist)
    elif no_slip_gas_fraction >= pure_hb_threshold:
        return _hb_holdup(fp, use_polynomial), 'hagedorn_brown'

    # Zone 3: Transition Window (Linear blend between the two)
    else:
        # Calculate how far we are into the transition zone (0.0 to 1.0)
        blend_factor = (no_slip_gas_fraction - pure_griffith_threshold) / max(pure_hb_threshold - pure_griffith_threshold, 1e-10)
        
        holdup_griffith = _griffith_holdup(fp)
        holdup_hb = _hb_holdup(fp, use_polynomial)
        
        # Weighted average
        blended_holdup = (1.0 - blend_factor) * holdup_griffith + (blend_factor) * holdup_hb
        
        return max(min(blended_holdup, 1.0), no_slip_liquid_fraction), 'transition_zone'


# ══════════════════════════════════════════════════════════════════════════════
#  PRESSURE GRADIENT
# ══════════════════════════════════════════════════════════════════════════════

def pressure_gradient_detailed(fp, roughness=0.0006, use_polynomial=False, friction_method="Churchill"):
    """
    Computes the total pressure gradient (dP/dz) for a specific pipe segment.
    Returns the total gradient (psi/ft) alongside a detailed dictionary of properties.
    """
    # ── 1. Unpack Fluid Properties for Readability ──────────────────────────
    liquid_density = fp['rho_l'] 
    gas_density = fp['rho_g']
    liquid_viscosity = fp['mu_l']
    gas_viscosity = fp['mu_g']
    mixture_velocity = fp['v_m']
    no_slip_liquid_fraction = fp['lam_l']
    pipe_id_inches = fp['d_in']
    pressure_psia = fp['p']
    superficial_gas_velocity = fp['v_sg']

    # ── 2. Calculate Liquid Holdup ──────────────────────────────────────────
    # This automatically determines if we are in Bubble flow or Slug/Mist
    liquid_holdup, flow_regime = _blended_holdup(fp, use_polynomial)

    # ── 3. Elevational (Hydrostatic) Gradient ───────────────────────────────
    # The mixture density is a volume-weighted average of liquid and gas densities
    mixture_density = liquid_density * liquid_holdup + gas_density * (1.0 - liquid_holdup)
    
    # Convert density (lb/ft³) to pressure gradient (psi/ft) by dividing by 144 in²
    elevational_gradient = mixture_density / 144.0

    # ── 4. Frictional Gradient ──────────────────────────────────────────────
    # For friction, Hagedorn-Brown uses a specific two-phase viscosity model:
    two_phase_viscosity = (liquid_viscosity ** liquid_holdup) * (gas_viscosity ** (1.0 - liquid_holdup))
    
    # And a "no-slip" density for the Reynolds number:
    no_slip_density = liquid_density * no_slip_liquid_fraction + gas_density * (1.0 - no_slip_liquid_fraction)

    # Calculate Reynolds Number (Re = 1488 * density * velocity * diameter / viscosity)
    pipe_id_ft = pipe_id_inches / 12.0
    if two_phase_viscosity > 0:
        reynolds_number = 1488.0 * no_slip_density * mixture_velocity * pipe_id_ft / two_phase_viscosity
    else:
        reynolds_number = 1e6 # Fallback for edge cases
        
    # Get the single-phase Darcy friction factor from the selected correlation
    base_friction_factor = _friction(reynolds_number, roughness, pipe_id_inches, friction_method)

    # Calculate the two-phase friction multiplier (s_multiplier)
    if liquid_holdup > 0:
        x_param = no_slip_liquid_fraction / (liquid_holdup ** 2)
    else:
        x_param = no_slip_liquid_fraction
        
    x_param = max(x_param, 1e-6)
    
    if 1.0 < x_param < 1.2:
        s_multiplier = math.log(2.2 * x_param - 1.2)
    else:
        log_x = math.log(x_param)
        denominator = -0.0523 + 3.182 * log_x - 0.8725 * (log_x ** 2) + 0.01853 * (log_x ** 4)
        if abs(denominator) > 1e-10:
            s_multiplier = log_x / denominator
        else:
            s_multiplier = 0.0
            
    two_phase_friction_factor = base_friction_factor * math.exp(s_multiplier)

    # Calculate the final frictional pressure drop (psi/ft)
    # Darcy-Weisbach: dp/dz = f_D * rho * v^2 / (2 * g_c * d * 144)
    # The 2.0 is in the DENOMINATOR because we are using the Darcy (Moody) friction factor.
    # (If using Fanning, the 2.0 would be in the numerator instead.)
    frictional_gradient = (two_phase_friction_factor * no_slip_density * (mixture_velocity ** 2)) / (2.0 * 32.174 * pipe_id_ft * 144.0)

    # ── 5. Acceleration (Kinetic) Term ──────────────────────────────────────
    # Accounts for pressure loss due to fluid accelerating (gas expanding as pressure drops)
    acceleration_term = (mixture_velocity * superficial_gas_velocity * mixture_density) / (32.174 * pressure_psia * 144.0)
    
    # Cap the acceleration term to prevent divide-by-zero errors in extreme conditions
    acceleration_term = max(min(acceleration_term, 0.99), 0.0)

    # ── 6. Total Pressure Gradient ──────────────────────────────────────────
    # Sum the elevation and friction, then divide by (1 - accel_term) to include kinetic effects
    total_gradient = (elevational_gradient + frictional_gradient) / (1.0 - acceleration_term)

    # Re-calculate the dimensionless groups to pass back out to the detailed table
    Nl, Nlv, Ngv, Nd = _dim_groups(fp)
    H = (Nlv / Ngv**0.575) * (pressure_psia / 14.7)**0.1 * _CNl(Nl) / Nd if (Ngv > 0 and Nd > 0) else 0.0
    psi = _psi((Ngv * Nlv**0.38) / Nd**2.14 if Nd > 0 else 0.0)

    details = {
        "flow_regime": flow_regime,
        "N_L": Nl,
        "CN_L": _CNl(Nl),
        "psi": psi,
        "H_L": liquid_holdup,
        "lambda_ns": no_slip_liquid_fraction,
        "lambda_g": 1.0 - no_slip_liquid_fraction,
        "rho_s": mixture_density,
        "rho_n": no_slip_density,
        "mu_n": two_phase_viscosity,
        "f_D": base_friction_factor,
        "frictional_gradient": frictional_gradient,
        "elevational_gradient": elevational_gradient,
        "accel_gradient": total_gradient * acceleration_term,
        "total_gradient": total_gradient,
        "V_m": mixture_velocity,
        "Re": reynolds_number,
        "Ek": acceleration_term
    }

    return total_gradient, details

# ══════════════════════════════════════════════════════════════════════════════
#  VLP CORRELATION DISPATCHER
# ══════════════════════════════════════════════════════════════════════════════

def _vlp_method_label(vlp_correlation: str = "Hagedorn-Brown") -> str:
    """
    Converts backend/UI VLP correlation key into display label.
    Hagedorn-Brown remains the default.
    """
    method = (vlp_correlation or "Hagedorn-Brown").strip().lower()

    if "beggs" in method or "brill" in method:
        return "Beggs-Brill"

    return "Modified Hagedorn-Brown"


def _is_beggs_brill(vlp_correlation: str = "Hagedorn-Brown") -> bool:
    """True only when Beggs-Brill is selected."""
    return _vlp_method_label(vlp_correlation) == "Beggs-Brill"


def _round_detail(details: dict, key: str, ndigits: int = 4):
    """
    Safe rounding helper for detailed table.
    Needed because Hagedorn-Brown and Beggs-Brill return different debug keys.
    """
    value = details.get(key)

    if value is None:
        return np.nan

    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return np.nan


def _calculate_vlp_gradient(
    fp: dict,
    roughness: float = 0.0006,
    use_polynomial: bool = False,
    friction_method: str = "Churchill",
    vlp_correlation: str = "Hagedorn-Brown",
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
    apply_payne: bool = True,
):
    """
    Connector dispatcher only.

    Hagedorn-Brown path:
        calls existing pressure_gradient_detailed() unchanged.

    Beggs-Brill path:
        calls backend/beggs_brill.py.
    """
    if _is_beggs_brill(vlp_correlation):
        return pressure_gradient_beggs_brill(
            fp=fp,
            inclination_deg=inclination_deg,
            flow_direction=flow_direction,
            friction_method=friction_method,
            apply_payne=apply_payne,
        )

    return pressure_gradient_detailed(
        fp,
        roughness=roughness,
        use_polynomial=use_polynomial,
        friction_method=friction_method,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  PRESSURE TRAVERSE  (Predictor-Corrector, WHP → BHP)
# ══════════════════════════════════════════════════════════════════════════════

def pressure_traverse(
    Pwh: float,
    q_oil: float,
    TVD: float,
    d_tubing: float,
    api: float,
    gas_sg: float,
    GOR: float,
    water_cut: float = 0.0,
    T_surface: float = 100.0,
    T_bottom: float = 200.0,
    water_sg: float = 1.0,
    roughness: float = 0.0006,
    n_segments: int = 50,
    pvt_method: str = "Standing",
    Psep: float = 100.0,
    Tsep: float = 75.0,
    salinity: float = 0.0,
    use_polynomial: bool = False,
    friction_method: str = "Churchill",
    vlp_correlation: str = "Hagedorn-Brown",
    MD: Optional[float] = None,
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
    apply_payne: bool = True,
    Pb: Optional[float] = None,
    apply_bo_undersat_correction: bool = False,
    Rs_manual: Optional[float] = None,
    Bo_manual: Optional[float] = None,
    mu_o_manual: Optional[float] = None,
) -> float:
    """March from wellhead to bottomhole using predictor-corrector."""
    if q_oil <= 0:
        T_avg = (T_surface + T_bottom) / 2.0
        P_est = Pwh + 0.3 * TVD

        pb_for_rs_s = Pb if (
            Pb is not None and Pb > 0
        ) else None

        if pvt_method == "Vasquez-Beggs":
            rs_calc_s, _ = calculate_rs_vasquez_beggs(
                P_est, T_avg, api, gas_sg,
                Psep=Psep,
                Tsep=Tsep,
                Pb=pb_for_rs_s,
            )
        else:
            rs_calc_s, _ = calculate_rs_standing(
                P_est, T_avg, api, gas_sg,
                Pb=pb_for_rs_s,
            )

        if Rs_manual is not None and Rs_manual > 0:
            Rs_s = min(float(Rs_manual), GOR)
        else:
            Rs_s = min(rs_calc_s, GOR)

        if (
            apply_bo_undersat_correction
            and Pb is not None
            and Pb > 0
            and P_est > Pb
        ):
            Rs_at_Pb_s = Rs_s

            if pvt_method == "Vasquez-Beggs":
                Bo_at_Pb_s, _ = calculate_bo_vasquez_beggs(
                    Rs_at_Pb_s, T_avg, api, gas_sg,
                    Psep=Psep,
                    Tsep=Tsep,
                )
            else:
                Bo_at_Pb_s, _ = calculate_bo_standing(
                    Rs_at_Pb_s, T_avg, api, gas_sg,
                )

            Bo_s, _ = calculate_bo_above_pb(
                Bo_at_Pb=Bo_at_Pb_s,
                P=P_est,
                Pb=Pb,
                Rs_at_Pb=Rs_at_Pb_s,
                T=T_avg,
                api=api,
                gas_sg=gas_sg,
            )

        else:
            if pvt_method == "Vasquez-Beggs":
                Bo_s, _ = calculate_bo_vasquez_beggs(
                    Rs_s, T_avg, api, gas_sg,
                    Psep=Psep,
                    Tsep=Tsep,
                )
            else:
                Bo_s, _ = calculate_bo_standing(
                    Rs_s, T_avg, api, gas_sg,
                )

        # Manual Bo still has highest priority.
        if Bo_manual is not None and Bo_manual > 0:
            Bo_s = float(Bo_manual)

        rho_o = _oil_density(api, Rs_s, Bo_s, gas_sg)
        rho_w = _water_density(water_sg)
        rho_avg = rho_o * (1.0 - water_cut) + rho_w * water_cut
        return Pwh + rho_avg * TVD / 144.0

    d_ft = d_tubing / 12.0
    A_pipe = PI / 4.0 * d_ft ** 2

    traverse_length = float(MD) if (_is_beggs_brill(vlp_correlation) and MD is not None) else TVD
    traverse_length = max(traverse_length, 1.0)
    dL = traverse_length / n_segments

    wc = max(min(water_cut, 0.999), 0.0)
    q_water = q_oil * wc / (1.0 - wc) if wc < 1.0 else 0.0
    dTdz = (T_bottom - T_surface) / traverse_length if traverse_length > 0 else 0.0

    P = Pwh
    for i in range(n_segments):
        T_top = T_surface + dTdz * i * dL
        T_bot = T_surface + dTdz * (i + 1) * dL
        T_mid = (T_top + T_bot) / 2.0

        fp_pred = _get_fluid_props(
            P, T_top, q_oil, q_water, GOR,
            d_tubing, d_ft, A_pipe, api, gas_sg, water_sg,
            pvt_method=pvt_method,
            Psep=Psep,
            Tsep=Tsep,
            salinity=salinity,
            Pb=Pb,
            apply_bo_undersat_correction=apply_bo_undersat_correction,
            Rs_manual=Rs_manual,
            Bo_manual=Bo_manual,
            mu_o_manual=mu_o_manual,
        )
        grad_pred, _ = _calculate_vlp_gradient(
            fp_pred,
            roughness=roughness,
            use_polynomial=use_polynomial,
            friction_method=friction_method,
            vlp_correlation=vlp_correlation,
            inclination_deg=inclination_deg,
            flow_direction=flow_direction,
            apply_payne=apply_payne,
        )
        dP_pred = grad_pred * dL
        P_pred = P + dP_pred

        # Numerical stability: cap predictor to max realistic gradient of 2 psi/ft.
        # At extreme flow rates, the acceleration term (Ek) can hit 0.99, causing
        # division by 0.01 and inflating the gradient by 100x. This makes P_pred
        # absurdly high, which tricks the corrector into thinking all gas is dissolved.
        P_pred = min(P_pred, P + 2.0 * dL)
        P_pred = max(P_pred, 1.0)

        P_avg = (P + P_pred) / 2.0
        fp_corr = _get_fluid_props(
            P_avg, T_mid, q_oil, q_water, GOR,
            d_tubing, d_ft, A_pipe, api, gas_sg, water_sg,
            pvt_method=pvt_method,
            Psep=Psep,
            Tsep=Tsep,
            salinity=salinity,
            Pb=Pb,
            apply_bo_undersat_correction=apply_bo_undersat_correction,
            Rs_manual=Rs_manual,
            Bo_manual=Bo_manual,
            mu_o_manual=mu_o_manual,
        )
        grad_corr, _ = _calculate_vlp_gradient(
            fp_corr,
            roughness=roughness,
            use_polynomial=use_polynomial,
            friction_method=friction_method,
            vlp_correlation=vlp_correlation,
            inclination_deg=inclination_deg,
            flow_direction=flow_direction,
            apply_payne=apply_payne,
        )
        dP_corr = grad_corr * dL

        P = P + dP_corr
        P = max(P, 1.0)

    return P


def detailed_pressure_traverse(
    Pwh: float, q_oil: float, TVD: float, d_tubing: float, api: float, gas_sg: float, GOR: float,
    water_cut: float = 0.0, T_surface: float = 100.0, T_bottom: float = 200.0, water_sg: float = 1.0,
    roughness: float = 0.0006, n_segments: int = 50, pvt_method: str = "Standing", Psep: float = 100.0,
    Tsep: float = 75.0, salinity: float = 0.0, use_polynomial: bool = False,
    friction_method: str = "Churchill",
    vlp_correlation: str = "Hagedorn-Brown",
    MD: Optional[float] = None,
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
    apply_payne: bool = True,
    Pb: Optional[float] = None,
    apply_bo_undersat_correction: bool = False,
    Rs_manual: Optional[float] = None,
    Bo_manual: Optional[float] = None,
    mu_o_manual: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    if q_oil <= 0:
        return pd.DataFrame()
    d_ft = d_tubing / 12.0
    A_pipe = PI / 4.0 * d_ft ** 2

    traverse_length = float(MD) if (_is_beggs_brill(vlp_correlation) and MD is not None) else TVD
    traverse_length = max(traverse_length, 1.0)
    dL = traverse_length / n_segments

    wc = max(min(water_cut, 0.999), 0.0)
    q_water = q_oil * wc / (1.0 - wc) if wc < 1.0 else 0.0
    dTdz = (T_bottom - T_surface) / traverse_length if traverse_length > 0 else 0.0
    P = Pwh
    for i in range(n_segments):
        T_top = T_surface + dTdz * i * dL
        T_bot = T_surface + dTdz * (i + 1) * dL
        T_mid = (T_top + T_bot) / 2.0
        
        fp_pred = _get_fluid_props(
            P, T_top, q_oil, q_water, GOR,
            d_tubing, d_ft, A_pipe, api, gas_sg, water_sg,
            pvt_method=pvt_method,
            Psep=Psep,
            Tsep=Tsep,
            salinity=salinity,
            Pb=Pb,
            apply_bo_undersat_correction=apply_bo_undersat_correction,
            Rs_manual=Rs_manual,
            Bo_manual=Bo_manual,
            mu_o_manual=mu_o_manual,
        )
        grad_pred, _ = _calculate_vlp_gradient(
            fp_pred,
            roughness=roughness,
            use_polynomial=use_polynomial,
            friction_method=friction_method,
            vlp_correlation=vlp_correlation,
            inclination_deg=inclination_deg,
            flow_direction=flow_direction,
            apply_payne=apply_payne,
        )
        dP_pred = grad_pred * dL
        P_pred = P + dP_pred

        # Numerical stability: cap predictor to max realistic gradient of 2 psi/ft.
        # At extreme flow rates, the acceleration term (Ek) can hit 0.99, causing
        # division by 0.01 and inflating the gradient by 100x. This makes P_pred
        # absurdly high, which tricks the corrector into thinking all gas is dissolved.
        P_pred = min(P_pred, P + 2.0 * dL)
        P_pred = max(P_pred, 1.0)

        P_avg = (P + P_pred) / 2.0
        fp_corr = _get_fluid_props(
            P_avg, T_mid, q_oil, q_water, GOR,
            d_tubing, d_ft, A_pipe, api, gas_sg, water_sg,
            pvt_method=pvt_method,
            Psep=Psep,
            Tsep=Tsep,
            salinity=salinity,
            Pb=Pb,
            apply_bo_undersat_correction=apply_bo_undersat_correction,
            Rs_manual=Rs_manual,
            Bo_manual=Bo_manual,
            mu_o_manual=mu_o_manual,
        )
        grad_corr, details = _calculate_vlp_gradient(
            fp_corr,
            roughness=roughness,
            use_polynomial=use_polynomial,
            friction_method=friction_method,
            vlp_correlation=vlp_correlation,
            inclination_deg=inclination_deg,
            flow_direction=flow_direction,
            apply_payne=apply_payne,
        )
        dP_corr = grad_corr * dL
        
        row = {
            "Depth / Length (ft)": round((i + 1) * dL, 1),
            "Pressure (psia)": round(P + dP_corr, 2),
            "VLP Correlation": details.get("method", _vlp_method_label(vlp_correlation)),
            "Flow Regime": str(details.get("flow_regime", "unknown")).capitalize().replace('_', ' '),

            # Common diagnostics
            "H_L (Holdup)": _round_detail(details, "H_L", 4),
            "λ_L / λ_ns": _round_detail(details, "lambda_ns", 4),
            "λ_g (Void Frac)": _round_detail(details, "lambda_g", 4),
            "V_m (ft/s)": _round_detail(details, "V_m", 2),
            "Re": _round_detail(details, "Re", 0),
            "Friction Grad (psi/ft)": _round_detail(details, "frictional_gradient", 4),
            "Elev Grad (psi/ft)": _round_detail(details, "elevational_gradient", 4),
            "Accel Grad (psi/ft)": _round_detail(details, "accel_gradient", 4),
            "Total Grad (psi/ft)": _round_detail(details, "total_gradient", 4),
            "Ek": _round_detail(details, "Ek", 5),

            # Hagedorn-Brown diagnostics
            "HB N_L": _round_detail(details, "N_L", 5),
            "HB CN_L": _round_detail(details, "CN_L", 5),
            "HB psi (ψ)": _round_detail(details, "psi", 4),

            # Beggs-Brill diagnostics
            "BB Inclination (deg)": _round_detail(details, "inclination_deg", 2),
            "BB Flow Direction": details.get("flow_direction", ""),
            "BB Nfr": _round_detail(details, "Nfr", 5),
            "BB Nlv": _round_detail(details, "Nlv", 5),
            "BB L1": _round_detail(details, "L1", 5),
            "BB L2": _round_detail(details, "L2", 5),
            "BB L3": _round_detail(details, "L3", 5),
            "BB L4": _round_detail(details, "L4", 5),
            "BB HL0": _round_detail(details, "HL0", 4),
            "BB C": _round_detail(details, "C", 4),
            "BB Payne Factor": _round_detail(details, "payne_factor", 4),
            "BB y": _round_detail(details, "y", 5),
            "BB S": _round_detail(details, "S", 5),
            "BB f_tp": _round_detail(details, "f_tp", 6),
        }
        rows.append(row)
        
        P = P + dP_corr
        P = max(P, 1.0)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  VLP CURVE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_vlp_curve(
    Pwh: float,
    TVD: float,
    d_tubing: float,
    api: float,
    gas_sg: float,
    GOR: float,
    water_cut: float = 0.0,
    T_surface: float = 100.0,
    T_bottom: float = 200.0,
    water_sg: float = 1.0,
    roughness: float = 0.0006,
    q_max: float = 3000.0,
    N: int = 100,
    n_segments: int = 50,
    pvt_method: str = "Standing",
    Psep: float = 100.0,
    Tsep: float = 75.0,
    salinity: float = 0.0,
    use_polynomial: bool = False,
    friction_method: str = "Churchill",
    vlp_correlation: str = "Hagedorn-Brown",
    MD: Optional[float] = None,
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
    apply_payne: bool = True,
    Pb: Optional[float] = None,
    apply_bo_undersat_correction: bool = False,
    Rs_manual: Optional[float] = None,
    Bo_manual: Optional[float] = None,
    mu_o_manual: Optional[float] = None,
) -> tuple[pd.DataFrame, dict, list[str]]:
    warnings: list[str] = []
    
    # Keep q = 0 only for static BHP calculation.
    # Do not include q = 0 in the flowing VLP curve.
    q_max = max(float(q_max), 1.0)
    N = max(int(N), 2)
    q_min = max(0.001 * q_max, 1.0)

    if q_min >= q_max:
        q_min = q_max / 100.0   

    q_values = np.linspace(q_min, q_max, N)
    pwf_values = np.empty(N)

    for idx, q in enumerate(q_values):
        pwf_values[idx] = pressure_traverse(
            Pwh=Pwh, q_oil=q, TVD=TVD, d_tubing=d_tubing,
            api=api, gas_sg=gas_sg, GOR=GOR,
            water_cut=water_cut,
            T_surface=T_surface, T_bottom=T_bottom,
            water_sg=water_sg, roughness=roughness,
            n_segments=n_segments,
            pvt_method=pvt_method, Psep=Psep, Tsep=Tsep, salinity=salinity,
            use_polynomial=use_polynomial, friction_method=friction_method,
            vlp_correlation=vlp_correlation,
            MD=MD,
            inclination_deg=inclination_deg,
            flow_direction=flow_direction,
            apply_payne=apply_payne,
            Pb=Pb,
            apply_bo_undersat_correction=apply_bo_undersat_correction,
            Rs_manual=Rs_manual,
            Bo_manual=Bo_manual,
            mu_o_manual=mu_o_manual,
        )

    Pwf_static = pressure_traverse(
        Pwh=Pwh,
        q_oil=0.0,
        TVD=TVD,
        d_tubing=d_tubing,
        api=api,
        gas_sg=gas_sg,
        GOR=GOR,
        water_cut=water_cut,
        T_surface=T_surface,
        T_bottom=T_bottom,
        water_sg=water_sg,
        roughness=roughness,
        n_segments=n_segments,
        pvt_method=pvt_method,
        Psep=Psep,
        Tsep=Tsep,
        salinity=salinity,
        use_polynomial=use_polynomial,
        friction_method=friction_method,
        vlp_correlation=vlp_correlation,
        MD=MD,
        inclination_deg=inclination_deg,
        flow_direction=flow_direction,
        apply_payne=apply_payne,
        Pb=Pb,
        apply_bo_undersat_correction=apply_bo_undersat_correction,
        Rs_manual=Rs_manual,
        Bo_manual=Bo_manual,
        mu_o_manual=mu_o_manual,
    )

    df = pd.DataFrame({
        # Rounded columns are kept for plotting/table display.
        "q (STB/day)": np.round(q_values, 2),
        "Pwf (psia)": np.round(pwf_values, 2),

        # Raw columns are used internally for accurate interpolation.
        "q_raw": q_values.astype(float),
        "Pwf_raw": pwf_values.astype(float),
    })



    key_results = {
        # Simplified static BHP estimate.
        # This uses an average liquid column only:
        #     Pwf_static = Pwh + rho_avg * TVD / 144
        # It is used only as a reference value, not for flowing VLP calculation.
        "Pwf_static": round(Pwf_static, 2),
        "Pwf_static_label": "Simplified Static BHP",
        "Pwf_static_note": (
            "Simplified liquid-column estimate using average oil/water density. "
            "Gas-column and static phase segregation effects are not modeled."
        ),
        "Pwh": Pwh,
        "TVD": TVD,
        "MD": MD if MD is not None else TVD,
        "d_tubing": d_tubing,
        "method": _vlp_method_label(vlp_correlation),
    }

    return df, key_results, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  OPERATING POINT  (IPR–VLP Intersection)
# ══════════════════════════════════════════════════════════════════════════════

def find_operating_point(
    df_ipr: pd.DataFrame,
    df_vlp: pd.DataFrame,
) -> dict:
    # Use raw VLP values when available.
    # This avoids operating-point error caused by rounded display columns.
    ipr_q = df_ipr["q_raw"].values.copy() if "q_raw" in df_ipr.columns else df_ipr["q (STB/day)"].values.copy()
    ipr_pwf = df_ipr["Pwf_raw"].values.copy() if "Pwf_raw" in df_ipr.columns else df_ipr["Pwf (psia)"].values.copy()

    vlp_q = df_vlp["q_raw"].values.copy() if "q_raw" in df_vlp.columns else df_vlp["q (STB/day)"].values.copy()
    vlp_pwf = df_vlp["Pwf_raw"].values.copy() if "Pwf_raw" in df_vlp.columns else df_vlp["Pwf (psia)"].values.copy()

    # Remove NaN / infinite points before interpolation
    ipr_mask = np.isfinite(ipr_q) & np.isfinite(ipr_pwf)
    vlp_mask = np.isfinite(vlp_q) & np.isfinite(vlp_pwf)

    ipr_q, ipr_pwf = ipr_q[ipr_mask], ipr_pwf[ipr_mask]
    vlp_q, vlp_pwf = vlp_q[vlp_mask], vlp_pwf[vlp_mask]

    if len(ipr_q) < 2 or len(vlp_q) < 2:
        return {
            "found": False,
            "q_op": None,
            "Pwf_op": None,
            "n_intersections": 0,
            "message": "Not enough valid IPR/VLP points after removing NaN values.",
        }

    ipr_order = np.argsort(ipr_q)
    ipr_q, ipr_pwf = ipr_q[ipr_order], ipr_pwf[ipr_order]

    vlp_order = np.argsort(vlp_q)
    vlp_q, vlp_pwf = vlp_q[vlp_order], vlp_pwf[vlp_order]

    q_lo = max(ipr_q[0], vlp_q[0])
    q_hi = min(ipr_q[-1], vlp_q[-1])

    if q_lo >= q_hi:
        return {
            "found": False, "q_op": None, "Pwf_op": None,
            "n_intersections": 0,
            "message": "IPR and VLP q ranges do not overlap.",
        }

    q_grid = np.linspace(q_lo, q_hi, 1000)
    pwf_ipr = np.interp(q_grid, ipr_q, ipr_pwf)
    pwf_vlp = np.interp(q_grid, vlp_q, vlp_pwf)

    diff = pwf_ipr - pwf_vlp
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    n_int = len(sign_changes)

    if n_int == 0:
            if np.all(diff > 0):
                msg = (
                    "No intersection within the generated rate range — "
                    "IPR remains above VLP. The operating point may lie at a "
                    "higher rate; increase the VLP maximum rate."
                )
            elif np.all(diff < 0):
                msg = (
                    "No intersection — VLP remains above IPR throughout the "
                    "generated rate range. The well cannot sustain flow under "
                    "the specified conditions."
                )
            else:
                msg = (
                    "No reliable intersection was detected. Increase curve "
                    "resolution and verify the IPR/VLP ranges."
                )
            return {
                "found": False, "q_op": None, "Pwf_op": None,
                "n_intersections": 0, "message": msg,
            }

    idx = sign_changes[-1]
    q1, q2 = q_grid[idx], q_grid[idx + 1]
    d1, d2 = diff[idx], diff[idx + 1]
    denom = d1 - d2
    frac = d1 / denom if abs(denom) > 1e-12 else 0.5
    q_op = q1 + frac * (q2 - q1)
    Pwf_op = float(np.interp(q_op, q_grid, pwf_ipr))

    msg = "Operating point found."
    if n_int > 1:
        msg += f" ({n_int} intersections detected; using rightmost / stable point.)"

    return {
        "found": True,
        "q_op": round(float(q_op), 2),
        "Pwf_op": round(float(Pwf_op), 2),
        "n_intersections": n_int,
        "message": msg,
    }
