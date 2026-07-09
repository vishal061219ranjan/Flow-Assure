"""
Flow Assure — PVT Correlations Module
======================================
Empirical correlations for oil, gas, and water properties.

All functions use **field units** unless otherwise noted:
- Pressure:    psia
- Temperature: °F
- GOR / Rs:    scf/STB
- Bo:          RB/STB
- Bg:          RB/scf
- Viscosity:   cP
- Density:     lb/ft³

Correlations Implemented
------------------------
Oil:
    • Solution GOR (Rs)       — Standing, Vasquez-Beggs
    • Bubble Point (Pb)       — Standing, Vasquez-Beggs
    • Oil FVF (Bo)            — Standing, Vasquez-Beggs
    • Dead Oil Viscosity      — Beggs-Robinson
    • Live Oil Viscosity      — Beggs-Robinson
    • Undersaturated Viscosity— Vasquez-Beggs

Gas:
    • Z-factor                — Dranchuk-Abou-Kassem (DAK)
    • Gas FVF (Bg)            — Direct calculation
    • Gas Viscosity (μg)      — Lee-Gonzalez-Eakin

Water (optional):
    • Bw                      — McCain
    • μw                      — McCain
"""

from __future__ import annotations

import math
from typing import Optional

#from plotly import api (CHANGE BACK TO NORMAL IF ERROR)

from backend.utils import (
    PVTResult,
    api_to_sg,
    fahrenheit_to_rankine,
    gas_molecular_weight,
    pseudo_critical_properties,
    safe_ln,
    safe_log10,
    R_UNIVERSAL,
)

# ══════════════════════════════════════════════════════════════════════════════
#  1. SOLUTION GAS-OIL RATIO  Rs
# ══════════════════════════════════════════════════════════════════════════════

def calculate_rs_standing(
    P: float,
    T: float,
    api: float,
    gas_sg: float,
    Pb: Optional[float] = None,
) -> tuple[float, list[str]]:
    """
    Solution gas-oil ratio using Standing correlation.

    Equation
    --------
    Rs = γg × [ (P / 18.2 + 1.4) × 10^x ]^1.2048
    where x = 0.0125 × API − 0.00091 × T

    Parameters
    ----------
    P      : Pressure, psia
    T      : Temperature, °F
    api    : Oil API gravity, °API
    gas_sg : Gas specific gravity (air = 1)
    Pb     : Bubble point pressure, psia (optional, to cap Rs)

    Returns
    -------
    tuple[float, list[str]]
        (Rs in scf/STB, warnings)
    """
    warnings: list[str] = []

    if P <= 0:
        warnings.append("Standing Rs: pressure must be positive. Returning Rs = 0.")
        return 0.0, warnings

    if gas_sg <= 0:
        warnings.append("Standing Rs: gas specific gravity must be positive. Returning Rs = 0.")
        return 0.0, warnings

    # Optional validity warnings
    if api < 16 or api > 58:
        warnings.append(f"Standing Rs: API={api:.2f} outside typical validity range [16, 58].")

    if T < 100 or T > 258:
        warnings.append(f"Standing Rs: T={T:.2f}°F outside typical validity range [100, 258].")

    # Step 1: Compute exponent
    x = 0.0125 * api - 0.00091 * T

    # Step 2: Compute Rs
    base = (P / 18.2 + 1.4) * (10.0 ** x)
    Rs = gas_sg * (base ** 1.2048)

    # Step 3: Cap Rs at bubble point Rs if P > Pb
    if Pb is not None and P > Pb:
        base_pb = (Pb / 18.2 + 1.4) * (10.0 ** x)
        Rs_pb = gas_sg * (base_pb ** 1.2048)
        Rs = Rs_pb

    return max(Rs, 0.0), warnings


def calculate_rs_vasquez_beggs(
    P: float,
    T: float,
    api: float,
    gas_sg: float,
    Psep: float = 100.0,
    Tsep: float = 75.0,
    Pb: Optional[float] = None,
) -> tuple[float, list[str]]:
    """
    Solution gas-oil ratio using **Vasquez-Beggs correlation.

    Equation
    --------
    Corrected gas gravity:
        γgs = γg × [1 + 5.912e-5 × API × Tsep × log10(Psep / 114.7)]

    For API ≤ 30:
        Rs = 0.0362 × γgs × P^1.0937 × exp(25.724 × API / (T + 459.67))

    For API > 30:
        Rs = 0.0178 × γgs × P^1.1870 × exp(23.931 × API / (T + 459.67))

    Parameters
    ----------
    P      : Pressure, psia
    T      : Temperature, °F
    api    : Oil API gravity, °API
    gas_sg : Gas specific gravity (air = 1)
    Psep   : Separator pressure, psig (default 100)
    Tsep   : Separator temperature, °F (default 75)
    Pb     : Bubble point pressure, psia (optional)

    Returns
    -------
    tuple[float, list[str]]
        (Rs in scf/STB, warnings)
    """
    warnings: list[str] = []

    # Correct gas gravity to 100-psig separator
    Psep_abs = Psep + 14.7  # convert psig to psia

    gamma_gs = gas_sg * (
        1.0 + 5.912e-5 * api * Tsep * safe_log10(Psep_abs / 114.7)
    )

    # Select coefficients
    if api <= 30:
        C1, C2, C3 = 0.0362, 1.0937, 25.7240
    else:
        C1, C2, C3 = 0.0178, 1.1870, 23.9310

    T_R = T + 459.67
    Rs = C1 * gamma_gs * (P ** C2) * math.exp(C3 * api / T_R)

    # Cap at bubble point
    if Pb is not None and P > Pb:
        Rs_pb = C1 * gamma_gs * (Pb ** C2) * math.exp(C3 * api / T_R)
        Rs = Rs_pb

    return max(Rs, 0.0), warnings


# ══════════════════════════════════════════════════════════════════════════════
#  2. BUBBLE POINT PRESSURE  Pb
# ══════════════════════════════════════════════════════════════════════════════

def calculate_pb_standing(
    Rs: float,
    T: float,
    api: float,
    gas_sg: float,
) -> tuple[float, list[str]]:
    """
    Bubble point pressure using Standing correlation (inverse of Rs).

    Equation
    --------
    Pb = 18.2 × [ (Rs / γg)^(1/1.2048) × 10^(−x) − 1.4 ]
    where x = 0.0125 × API − 0.00091 × T

    Parameters
    ----------
    Rs     : Solution gas-oil ratio at Pb, scf/STB
    T      : Temperature, °F
    api    : Oil API gravity, °API
    gas_sg : Gas specific gravity (air = 1)

    Returns
    -------
    tuple[float, list[str]]
        (Pb in psia, warnings)
    """
    warnings: list[str] = []

    if gas_sg <= 0:
        warnings.append("Gas SG must be positive for Standing Pb.")
        return 0.0, warnings

    x = 0.0125 * api - 0.00091 * T
    ratio = (Rs / gas_sg) ** (1.0 / 1.2048)
    Pb = 18.2 * (ratio * (10.0 ** (-x)) - 1.4)

    if Pb < 0:
        warnings.append(f"Standing Pb calculation yielded negative value ({Pb:.1f}); clamped to 14.7 psia.")
        Pb = 14.7

    return Pb, warnings


def calculate_pb_vasquez_beggs(
    Rs: float,
    T: float,
    api: float,
    gas_sg: float,
    Psep: float = 100.0,
    Tsep: float = 75.0,
) -> tuple[float, list[str]]:
    """
    Bubble point pressure using Vasquez-Beggs correlation (inverse of Rs).

    Equation
    --------
    Rearranged from VB Rs equation:
        Pb = [ Rs / (C1 × γgs × exp(C3 × API / (T+459.67))) ]^(1/C2)
    """
    warnings: list[str] = []

    Psep_abs = Psep + 14.7  # convert psig to psia

    gamma_gs = gas_sg * (
        1.0 + 5.912e-5 * api * Tsep * safe_log10(Psep_abs / 114.7)
    )

    if api <= 30:
        C1, C2, C3 = 0.0362, 1.0937, 25.7240
    else:
        C1, C2, C3 = 0.0178, 1.1870, 23.9310

    T_R = T + 459.67
    denom = C1 * gamma_gs * math.exp(C3 * api / T_R)

    if denom <= 0:
        warnings.append("VB Pb: denominator ≤ 0; cannot compute.")
        return 0.0, warnings

    Pb = (Rs / denom) ** (1.0 / C2)

    if Pb < 0:
        warnings.append(f"VB Pb yielded negative value; clamped to 14.7.")
        Pb = 14.7

    return Pb, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  3. OIL FORMATION VOLUME FACTOR  Bo
# ══════════════════════════════════════════════════════════════════════════════

def calculate_bo_standing(
    Rs: float,
    T: float,
    api: float,
    gas_sg: float,
) -> tuple[float, list[str]]:
    """
    Oil formation volume factor using Standing correlation.

    Equation
    --------
    F  = Rs × (γg / γo)^0.5 + 1.25 × T
    Bo = 0.972 + 1.47e-4 × F^1.175

    where γo = 141.5 / (API + 131.5)

    Parameters
    ----------
    Rs     : Solution GOR at the pressure of interest, scf/STB
    T      : Temperature, °F
    api    : Oil API gravity
    gas_sg : Gas specific gravity

    Returns
    -------
    tuple[float, list[str]]
        (Bo in RB/STB, warnings)
    """
    warnings: list[str] = []

    gamma_o = api_to_sg(api)

    if gamma_o <= 0:
        warnings.append("Oil SG must be positive.")
        return 1.0, warnings

    # Compute F parameter
    F = Rs * math.sqrt(gas_sg / gamma_o) + 1.25 * T

    # Compute Bo
    Bo = 0.972 + 1.47e-4 * (F ** 1.175)

    if Bo < 1.0:
        warnings.append(f"Standing Bo={Bo:.4f} < 1.0; physically unrealistic, clamped to 1.0.")
        Bo = 1.0

    return Bo, warnings


def calculate_bo_vasquez_beggs(
    Rs: float,
    T: float,
    api: float,
    gas_sg: float,
    Psep: float = 100.0,
    Tsep: float = 75.0,
) -> tuple[float, list[str]]:
    """
    Oil formation volume factor using Vasquez-Beggscorrelation.

    Equation
    --------
    For API ≤ 30:
        Bo = 1 + 4.677e-4×Rs + 1.751e-5×(T-60)×(API/γgs) − 1.811e-8×Rs×(T-60)×(API/γgs)
    For API > 30:
        Bo = 1 + 4.670e-4×Rs + 1.100e-5×(T-60)×(API/γgs) + 1.337e-9×Rs×(T-60)×(API/γgs)
    """
    warnings: list[str] = []

    Psep_abs = Psep + 14.7  # convert psig to psia

    gamma_gs = gas_sg * (
        1.0 + 5.912e-5 * api * Tsep * safe_log10(Psep_abs / 114.7)
    )

    if gamma_gs <= 0:
        gamma_gs = gas_sg  # fallback

    if api <= 30:
        C1, C2, C3 = 4.677e-4, 1.751e-5, -1.811e-8
    else:
        C1, C2, C3 = 4.670e-4, 1.100e-5, 1.337e-9

    term = (T - 60.0) * (api / gamma_gs)
    Bo = 1.0 + C1 * Rs + C2 * term + C3 * Rs * term

    if Bo < 1.0:
        warnings.append(f"VB Bo={Bo:.4f} < 1.0; clamped to 1.0.")
        Bo = 1.0

    return Bo, warnings

#=======================================================
#                Compressibility calculation
#=======================================================

def calculate_co_vasquez_beggs(
    P: float,
    T: float,
    api: float,
    gas_sg: float,
    Rs_at_Pb: float,
    default_co: float = 10.0e-6,
) -> tuple[float, list[str]]:
    """
    Equation
    --------
    co = [-1433 + 5*Rsb + 17.2*T - 1180*γg + 12.61*API] / (1e5 * P)

    Parameters
    ----------
    P        : Current pressure, psia
    T        : Reservoir temperature, °F
    api      : Oil API gravity, °API
    gas_sg   : Gas specific gravity, air = 1
    Rs_at_Pb : Solution GOR at bubble point, scf/STB
    default_co : fallback oil compressibility, 1/psi

    Returns
    -------
    tuple[float, list[str]]
        (co in 1/psi, warnings)
    """
    warnings: list[str] = []

    if P <= 0:
        warnings.append(
            f"Vasquez-Beggs co: P={P:.2f} psia is invalid. "
            f"Using default co={default_co:.2e} 1/psi."
        )
        return default_co, warnings

    numerator = (
        -1433.0
        + 5.0 * Rs_at_Pb
        + 17.2 * T
        - 1180.0 * gas_sg
        + 12.61 * api
    )

    co = numerator / (1.0e5 * P)

    if co <= 0 or not math.isfinite(co):
        warnings.append(
            f"Vasquez-Beggs co calculated as {co:.3e} 1/psi, which is non-physical. "
            f"Using default co={default_co:.2e} 1/psi."
        )
        co = default_co

    return co, warnings

def calculate_bo_above_pb(
    Bo_at_Pb: float,
    P: float,
    Pb: float,
    Rs_at_Pb: float,
    T: float,
    api: float,
    gas_sg: float,
    default_co: float = 10.0e-6,
) -> tuple[float, list[str]]:
    """
    Oil FVF above bubble point using Vasquez-Beggs oil compressibility.

    Equation
    --------
    Bo(P) = Bo(Pb) × exp[-co × (P − Pb)]

    Parameters
    ----------
    Bo_at_Pb : Bo at bubble point, RB/STB
    P        : Current pressure, psia
    Pb       : Bubble point pressure, psia
    Rs_at_Pb : Solution GOR at bubble point, scf/STB
    T        : Reservoir temperature, °F
    api      : Oil API gravity, °API
    gas_sg   : Gas specific gravity, air = 1
    default_co : fallback oil compressibility, 1/psi

    Returns
    -------
    tuple[float, list[str]]
        (Bo at P, warnings)
    """
    warnings: list[str] = []

    if Bo_at_Pb <= 0:
        warnings.append("Bo_at_Pb must be positive. Using Bo = 1.0 RB/STB.")
        return 1.0, warnings

    if Pb <= 0:
        warnings.append("Pb must be positive for undersaturated Bo correction. Returning Bo_at_Pb.")
        return Bo_at_Pb, warnings

    if P <= Pb:
        warnings.append(
            f"P={P:.2f} psia is not above Pb={Pb:.2f} psia. "
            "Undersaturated Bo correction not applied."
        )
        return Bo_at_Pb, warnings

    co, co_warnings = calculate_co_vasquez_beggs(
        P=P,
        T=T,
        api=api,
        gas_sg=gas_sg,
        Rs_at_Pb=Rs_at_Pb,
        default_co=default_co,
    )
    warnings.extend(co_warnings)

    Bo = Bo_at_Pb * math.exp(-co * (P - Pb))

    if Bo < 1.0:
        warnings.append(f"Bo above Pb = {Bo:.4f} < 1.0; clamped to 1.0.")
        Bo = 1.0

    return Bo, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  4. OIL VISCOSITY
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  4. OIL VISCOSITY
#     Dead oil      : Egbogah-type correlation
#     Saturated oil : Beggs-Robinson
#     Undersat oil  : Vasquez-Beggs
# ══════════════════════════════════════════════════════════════════════════════

def calculate_dead_oil_viscosity_egbogah(
    api: float,
    T: float,
) -> tuple[float, list[str]]:
    """
    Dead oil viscosity using Egbogah-type correlation.

    Equation
    --------
    μod = 10^[10^(1.8653 − 0.025086*API − 0.5644*log10(T))] − 1

    Parameters
    ----------
    api : Oil API gravity, °API
    T   : Temperature, °F

    Returns
    -------
    tuple[float, list[str]]
        (dead oil viscosity μod in cP, warnings)
    """
    warnings: list[str] = []

    if api <= 0:
        warnings.append(
            f"Egbogah dead oil viscosity: API={api:.2f} is invalid. "
            "Using API=35 as fallback."
        )
        api = 35.0

    if T <= 0:
        warnings.append(
            f"Egbogah dead oil viscosity: T={T:.2f}°F is invalid. "
            "Using T=180°F as fallback."
        )
        T = 180.0

    if api < 10 or api > 70:
        warnings.append(
            f"Egbogah dead oil viscosity: API={api:.2f} is outside typical range [10, 70]."
        )

    if T < 60 or T > 300:
        warnings.append(
            f"Egbogah dead oil viscosity: T={T:.2f}°F is outside typical range [60, 300]."
        )

    exponent_inner = 1.8653 - 0.025086 * api - 0.5644 * math.log10(T)

    try:
        mu_dead = 10.0 ** (10.0 ** exponent_inner) - 1.0
    except OverflowError:
        warnings.append(
            "Egbogah dead oil viscosity overflowed. "
            "Using fallback μod = 1.0 cP."
        )
        mu_dead = 1.0

    if mu_dead <= 0 or not math.isfinite(mu_dead):
        warnings.append(
            f"Egbogah dead oil viscosity calculated as {mu_dead:.4g} cP, "
            "which is non-physical. Using fallback μod = 1.0 cP."
        )
        mu_dead = 1.0

    return mu_dead, warnings


def calculate_dead_oil_viscosity_beggs_robinson(
    api: float,
    T: float,
) -> tuple[float, list[str]]:
    """
    Backward-compatible wrapper.

    Earlier code called this function name for dead oil viscosity.
    To avoid breaking pvt.py and vlp.py imports, this function now uses
    Egbogah-type dead oil viscosity internally.
    """
    return calculate_dead_oil_viscosity_egbogah(api, T)


def calculate_saturated_oil_viscosity_beggs_robinson(
    mu_dead: float,
    Rs: float,
) -> tuple[float, list[str]]:
    """
    Saturated oil viscosity using Beggs-Robinson correlation.

    Equation
    --------
    A  = 10.715 × (Rs + 100)^(-0.515)
    B  = 5.44 × (Rs + 150)^(-0.338)
    μo = A × μod^B

    Parameters
    ----------
    mu_dead : Dead oil viscosity μod, cP
    Rs      : Solution GOR at pressure P, scf/STB

    Returns
    -------
    tuple[float, list[str]]
        (saturated oil viscosity μo in cP, warnings)
    """
    warnings: list[str] = []

    if mu_dead <= 0 or not math.isfinite(mu_dead):
        warnings.append(
            f"Beggs-Robinson saturated viscosity: μod={mu_dead:.4g} cP is invalid. "
            "Using μod = 1.0 cP."
        )
        mu_dead = 1.0

    if Rs < 0:
        warnings.append(
            f"Beggs-Robinson saturated viscosity: Rs={Rs:.2f} scf/STB is negative. "
            "Using Rs = 0."
        )
        Rs = 0.0

    A = 10.715 * ((Rs + 100.0) ** (-0.515))
    B = 5.44 * ((Rs + 150.0) ** (-0.338))

    mu_o = A * (mu_dead ** B)

    if mu_o <= 0 or not math.isfinite(mu_o):
        warnings.append(
            f"Beggs-Robinson saturated viscosity calculated as {mu_o:.4g} cP, "
            "which is non-physical. Using fallback μo = 1.0 cP."
        )
        mu_o = 1.0

    return mu_o, warnings


def calculate_live_oil_viscosity_beggs_robinson(
    mu_dead: float,
    Rs: float,
) -> tuple[float, list[str]]:
    """
    Backward-compatible wrapper.

    Earlier code used this function name for live/saturated oil viscosity.
    It now calls calculate_saturated_oil_viscosity_beggs_robinson().
    """
    return calculate_saturated_oil_viscosity_beggs_robinson(mu_dead, Rs)


def calculate_undersaturated_oil_viscosity(
    mu_ob: float,
    P: float,
    Pb: float,
) -> tuple[float, list[str]]:
    """
    Undersaturated oil viscosity using Vasquez-Beggs correction.

    Equation
    --------
    m  = 2.6 × P^1.187 × exp(−11.513 − 8.98e−5 × P)
    μo = μob × (P / Pb)^m

    Parameters
    ----------
    mu_ob : Oil viscosity at bubble point pressure, cP
    P     : Current pressure, psia
    Pb    : Bubble point pressure, psia

    Returns
    -------
    tuple[float, list[str]]
        (undersaturated oil viscosity μo in cP, warnings)
    """
    warnings: list[str] = []

    if mu_ob <= 0 or not math.isfinite(mu_ob):
        warnings.append(
            f"Vasquez-Beggs undersaturated viscosity: μob={mu_ob:.4g} cP is invalid. "
            "Using μob = 1.0 cP."
        )
        mu_ob = 1.0

    if Pb <= 0:
        warnings.append(
            "Vasquez-Beggs undersaturated viscosity: Pb must be positive. "
            "Returning μob."
        )
        return mu_ob, warnings

    if P <= Pb:
        warnings.append(
            f"P={P:.2f} psia is not above Pb={Pb:.2f} psia. "
            "Undersaturated viscosity correction not applied."
        )
        return mu_ob, warnings

    m = 2.6 * (P ** 1.187) * math.exp(-11.513 - 8.98e-5 * P)

    mu_o = mu_ob * ((P / Pb) ** m)

    if mu_o <= 0 or not math.isfinite(mu_o):
        warnings.append(
            f"Vasquez-Beggs undersaturated viscosity calculated as {mu_o:.4g} cP, "
            "which is non-physical. Returning μob."
        )
        mu_o = mu_ob

    return mu_o, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  5. GAS Z-FACTOR  (Dranchuk-Abou-Kassem, 1975)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_z_factor_dak(
    P: float,
    T: float,
    gas_sg: float,
    max_iter: int = 100,
    tol: float = 1.0e-10,
) -> tuple[float, list[str]]:
    """
    Gas compressibility factor using Dranchuk-Abou-Kassem .

    This version solves the DAK equation in terms of reduced density rho_r
    using a safeguarded Newton-bisection method.

    DAK relationship
    ----------------
    rho_r = 0.27 * Ppr / (Z * Tpr)

    Therefore:
    Z = 0.27 * Ppr / (rho_r * Tpr)

    The DAK EOS also gives:
    Z = f(rho_r, Tpr)

    So we solve:
    f(rho_r, Tpr) - 0.27*Ppr/(rho_r*Tpr) = 0

    Parameters
    ----------
    P      : Pressure, psia
    T      : Temperature, °F
    gas_sg : Gas specific gravity, air = 1
    max_iter : Maximum Newton-bisection iterations
    tol      : Convergence tolerance

    Returns
    -------
    tuple[float, list[str]]
        (Z-factor dimensionless, warnings)
    """
    warnings: list[str] = []

    if P <= 0:
        return 1.0, ["Pressure ≤ 0; Z = 1.0 used as ideal-gas fallback."]

    if gas_sg <= 0:
        return 1.0, ["Gas specific gravity ≤ 0; Z = 1.0 used as fallback."]

    # Pseudo-critical properties
    Tpc, Ppc = pseudo_critical_properties(gas_sg)

    if Tpc <= 0 or Ppc <= 0:
        return 1.0, ["Invalid pseudo-critical properties; Z = 1.0 used as fallback."]

    T_R = fahrenheit_to_rankine(T)
    if T_R <= 0:
        return 1.0, ["Temperature in Rankine ≤ 0; Z = 1.0 used as fallback."]

    Tpr = T_R / Tpc
    Ppr = P / Ppc

    # DAK validity warnings
    if Tpr < 1.05 or Tpr > 3.0:
        warnings.append(
            f"DAK Z: Tpr={Tpr:.3f} outside recommended range [1.05, 3.0]."
        )

    if Ppr < 0.2 or Ppr > 30.0:
        warnings.append(
            f"DAK Z: Ppr={Ppr:.3f} outside recommended range [0.2, 30.0]."
        )

    # DAK coefficients
    A1, A2, A3 = 0.3265, -1.0700, -0.5339
    A4, A5 = 0.01569, -0.05165
    A6, A7, A8 = 0.5475, -0.7361, 0.1844
    A9, A10, A11 = 0.1056, 0.6134, 0.7210

    R2 = 0.27 * Ppr / Tpr

    def z_from_rho(rho_r: float) -> float:
        """
        DAK EOS explicit Z as function of reduced density rho_r.
        """
        rr2 = rho_r * rho_r
        rr5 = rho_r ** 5

        term1 = (
            A1
            + A2 / Tpr
            + A3 / (Tpr ** 3)
            + A4 / (Tpr ** 4)
            + A5 / (Tpr ** 5)
        ) * rho_r

        term2 = (
            A6
            + A7 / Tpr
            + A8 / (Tpr ** 2)
        ) * rr2

        term3 = -A9 * (
            A7 / Tpr
            + A8 / (Tpr ** 2)
        ) * rr5

        term4 = (
            A10
            * (1.0 + A11 * rr2)
            * (rr2 / (Tpr ** 3))
            * math.exp(-A11 * rr2)
        )

        return 1.0 + term1 + term2 + term3 + term4

    def dz_drho(rho_r: float) -> float:
        """
        Derivative of DAK EOS Z with respect to reduced density rho_r.
        Used for Newton step.
        """
        a = (
            A1
            + A2 / Tpr
            + A3 / (Tpr ** 3)
            + A4 / (Tpr ** 4)
            + A5 / (Tpr ** 5)
        )

        b = (
            A6
            + A7 / Tpr
            + A8 / (Tpr ** 2)
        )

        c = A9 * (
            A7 / Tpr
            + A8 / (Tpr ** 2)
        )

        d = A10 / (Tpr ** 3)
        k = A11

        exp_term = math.exp(-k * rho_r * rho_r)

        # derivative of:
        # rho^2 * (1 + k*rho^2) * exp(-k*rho^2)
        d_special = exp_term * (
            2.0 * rho_r
            + 2.0 * k * (rho_r ** 3)
            - 2.0 * (k ** 2) * (rho_r ** 5)
        )

        return (
            a
            + 2.0 * b * rho_r
            - 5.0 * c * (rho_r ** 4)
            + d * d_special
        )

    def residual(rho_r: float) -> float:
        """
        DAK residual:
        z_from_rho(rho_r) - R2/rho_r = 0
        """
        return z_from_rho(rho_r) - (R2 / rho_r)

    def residual_derivative(rho_r: float) -> float:
        """
        Derivative of residual.
        d/drho [z_from_rho - R2/rho] = dz/drho + R2/rho^2
        """
        return dz_drho(rho_r) + R2 / (rho_r ** 2)

    # ------------------------------------------------------------------
    # Bracket the physical positive reduced-density root
    # ------------------------------------------------------------------

    rho_low = 1.0e-10
    rho_high = 5.0

    f_low = residual(rho_low)
    f_high = residual(rho_high)

    # Expand upper bracket if needed
    while f_low * f_high > 0 and rho_high < 100.0:
        rho_high *= 2.0
        f_high = residual(rho_high)

    if f_low * f_high > 0:
        warnings.append(
            "DAK Z: could not bracket reduced-density root. "
            "Returning Z = 1.0 fallback."
        )
        return 1.0, warnings

    # Initial guess from ideal gas Z ≈ 1
    rho_r = max(min(R2, rho_high), rho_low)

    converged = False

    for _ in range(max_iter):
        f = residual(rho_r)

        if abs(f) < tol:
            converged = True
            break

        # Maintain bracket
        if f_low * f < 0:
            rho_high = rho_r
            f_high = f
        else:
            rho_low = rho_r
            f_low = f

        # Newton step
        df = residual_derivative(rho_r)

        if df != 0 and math.isfinite(df):
            rho_new = rho_r - f / df
        else:
            rho_new = 0.5 * (rho_low + rho_high)

        # Safeguard: if Newton goes outside bracket, use bisection
        if (
            rho_new <= rho_low
            or rho_new >= rho_high
            or not math.isfinite(rho_new)
        ):
            rho_new = 0.5 * (rho_low + rho_high)

        if abs(rho_new - rho_r) < tol:
            rho_r = rho_new
            converged = True
            break

        rho_r = rho_new

    if not converged:
        warnings.append("DAK Z-factor did not fully converge; returning last iterate.")

    if rho_r <= 0 or not math.isfinite(rho_r):
        warnings.append("DAK Z: invalid reduced density; Z = 1.0 fallback used.")
        return 1.0, warnings

    Z = R2 / rho_r

    if Z <= 0 or not math.isfinite(Z):
        warnings.append("DAK Z: non-physical Z calculated; Z = 1.0 fallback used.")
        Z = 1.0

    if rho_r > 3.0:
        warnings.append(
            f"DAK Z: final reduced density rho_r={rho_r:.3f} is very high; "
            "result may be extrapolated."
        )

    return Z, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  6. GAS FORMATION VOLUME FACTOR  Bg
# ══════════════════════════════════════════════════════════════════════════════

def calculate_bg(
    P: float,
    T: float,
    Z: float,
) -> tuple[float, list[str]]:
    """
    Gas formation volume factor.

    Equation
    --------
    Bg = 0.005035 × Z × (T + 460) / P     [RB/scf]

    Parameters
    ----------
    P : Pressure, psia
    T : Temperature, °F
    Z : Gas compressibility factor

    Returns
    -------
    tuple[float, list[str]]
        (Bg in RB/scf, warnings)
    """
    warnings: list[str] = []
    if P <= 0:
        warnings.append("P ≤ 0; cannot compute Bg.")
        return 0.0, warnings

    Bg = 0.005035 * Z * (T + 460) / P
    return Bg, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  7. GAS VISCOSITY  (Lee-Gonzalez-Eakin, 1966)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_gas_viscosity_lge(
    P: float,
    T: float,
    gas_sg: float,
    Z: float,
) -> tuple[float, list[str]]:
    """
    Gas viscosity using Lee-Gonzalez-Eakin correlation.

    Equation
    --------
    Mg  = 28.97 × γg
    ρg  = P × Mg / (Z × R × (T + 459.67))        [lb/ft³]
    K   = (9.4 + 0.02×Mg) × (T+459.67)^1.5 / (209 + 19×Mg + (T+459.67))
    X   = 3.5 + 986/(T+459.67) + 0.01×Mg
    Y   = 2.4 − 0.2×X
    μg  = 1e-4 × K × exp(X × (ρg/62.428)^Y)     [cP]

    Parameters
    ----------
    P      : Pressure, psia
    T      : Temperature, °F
    gas_sg : Gas specific gravity
    Z      : Gas compressibility factor

    Returns
    -------
    tuple[float, list[str]]
        (μg in cP, warnings)
    """
    warnings: list[str] = []

    Mg = gas_molecular_weight(gas_sg)
    T_R = T + 459.67

    if Z <= 0 or T_R <= 0 or P <= 0:
        warnings.append("Invalid inputs for LGE gas viscosity.")
        return 0.01, warnings

    # Gas density in lb/ft³
    rho_g = P * Mg / (Z * R_UNIVERSAL * T_R)

    K = ((9.4 + 0.02 * Mg) * (T_R ** 1.5)) / (209.0 + 19.0 * Mg + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * Mg
    Y = 2.4 - 0.2 * X

    mu_g = 1.0e-4 * K * math.exp(X * ((rho_g / 62.428) ** Y)) #We used rho_g/62.428 it is in g/cm³

    if mu_g <= 0:
        warnings.append("μg ≤ 0; clamped to 0.001 cP.")
        mu_g = 0.001

    return mu_g, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  8. WATER PROPERTIES (Optional — McCain)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_bw(P: float, T: float) -> tuple[float, list[str]]:
    """
    Water formation volume factor using **McCain** correlation.

    Equation
    --------
    ΔVwT = −1.0001e-2 + 1.33391e-4×T + 5.50654e-7×T²
    ΔVwp = −1.95301e-9×P×T − 1.72834e-13×P²×T − 3.58922e-7×P − 2.25341e-10×P²
    Bw   = (1 + ΔVwp) × (1 + ΔVwT)
    """
    warnings: list[str] = []
    dVwT = -1.0001e-2 + 1.33391e-4 * T + 5.50654e-7 * T ** 2
    dVwp = (-1.95301e-9 * P * T - 1.72834e-13 * P ** 2 * T
            - 3.58922e-7 * P - 2.25341e-10 * P ** 2)
    Bw = (1.0 + dVwp) * (1.0 + dVwT)
    if Bw < 0.9:
        warnings.append(f"Bw = {Bw:.4f} seems low.")
    return Bw, warnings


def calculate_muw(
    P: float,
    T: float,
    salinity: float = 0.0,
) -> tuple[float, list[str]]:
    """
    Water viscosity using McCain correlation.

    Equations
    ---------
    μw1 = A × T^(-B)

    A = 109.574 − 8.40564*Cw + 0.313314*Cw² + 0.00872213*Cw³

    B = −1.12166 + 2.63951e−2*Cw − 6.79461e−4*Cw²
        − 5.47119e−5*Cw³ + 1.55586e−6*Cw⁴

    μw = μw1 × (0.9994 + 4.0295e−5*P + 3.1062e−9*P²)

    Parameters
    ----------
    P        : Pressure, psia
    T        : Temperature, °F
    salinity : Water salinity Cw, wt% solids / wt% NaCl

    Returns
    -------
    tuple[float, list[str]]
        (water viscosity μw in cP, warnings)
    """
    warnings: list[str] = []

    if T <= 0:
        warnings.append("T must be > 0 for μw.")
        return 1.0, warnings

    if P < 0:
        warnings.append("P cannot be negative for μw. Using P = 0.")
        P = 0.0

    Cw = salinity

    A = (
        109.574
        - 8.40564 * Cw
        + 0.313314 * Cw ** 2
        + 8.72213e-3 * Cw ** 3
    )

    B = (
        -1.12166
        + 2.63951e-2 * Cw
        - 6.79461e-4 * Cw ** 2
        - 5.47119e-5 * Cw ** 3
        + 1.55586e-6 * Cw ** 4
    )

    # Water viscosity at atmospheric pressure
    mu_w1 = A * (T ** B)

    # Pressure correction
    pressure_factor = 0.9994 + 4.0295e-5 * P + 3.1062e-9 * P ** 2

    mu_w = mu_w1 * pressure_factor

    if mu_w <= 0:
        warnings.append("μw ≤ 0; defaulted to 0.5 cP.")
        mu_w = 0.5

    return mu_w, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  9. MASTER PVT CALCULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def calculate_all_pvt(inputs: dict, pvt_method: str = "Standing") -> dict:
    """
    Calculate all PVT properties in the correct dependency order.

    This is the main entry point called by the Streamlit app.

    Parameters
    ----------
    inputs : dict
        Dictionary of user-provided (or default) input values.
        Keys may include: Pr, Pwf_test, Pb, T, api, gas_sg, GOR,
        Bo_manual, Rs_manual, mu_o_manual, Psep, Tsep, salinity
    pvt_method : str
        "Standing" or "Vasquez-Beggs"

    Returns
    -------
    dict with keys:
        "results"  : dict of PVTResult objects keyed by symbol
        "warnings" : list[str]  (aggregated)
        "values"   : dict of {symbol: float}  for easy downstream use
    """
    results: dict[str, PVTResult] = {}
    all_warnings: list[str] = []
    values: dict[str, float] = {}

    Pr = inputs.get("Pr")
    T = inputs.get("T")
    api = inputs.get("api")
    gas_sg = inputs.get("gas_sg", 0.75)
    Psep = inputs.get("Psep", 100.0)
    Tsep = inputs.get("Tsep", 75.0)
    GOR = inputs.get("GOR")
    salinity = inputs.get("salinity", 0.0)

    # User manual overrides
    Pb_manual = inputs.get("Pb_manual")
    Rs_manual = inputs.get("Rs_manual")
    Bo_manual = inputs.get("Bo_manual")
    mu_o_manual = inputs.get("mu_o_manual")

    # ── Step 1: Bubble Point Pressure (Pb) ────────────────────────────────
    Pb_val = None
    if Pb_manual is not None and Pb_manual > 0:
        Pb_val = Pb_manual
        results["Pb"] = PVTResult("Bubble Point Pressure", "Pb", Pb_val, "psia", "User Input")
    else:
        # Need Rs (or GOR as proxy) to estimate Pb
        Rs_for_Pb = Rs_manual if (Rs_manual is not None and Rs_manual > 0) else GOR
        if Rs_for_Pb is not None and Rs_for_Pb > 0 and T is not None and api is not None:
            if pvt_method == "Standing":
                Pb_val, w = calculate_pb_standing(Rs_for_Pb, T, api, gas_sg)
                src = "Standing Correlation"
            else:
                Pb_val, w = calculate_pb_vasquez_beggs(Rs_for_Pb, T, api, gas_sg, Psep, Tsep)
                src = "Vasquez-Beggs Correlation"
            all_warnings.extend(w)
            results["Pb"] = PVTResult("Bubble Point Pressure", "Pb", Pb_val, "psia", src, w)
        else:
            msg = "Cannot estimate Pb: need Rs (or GOR), T, and API."
            all_warnings.append(msg)
            results["Pb"] = PVTResult("Bubble Point Pressure", "Pb", None, "psia", "N/A", [msg])

    # ── Step 2: Solution GOR (Rs) at reservoir pressure ───────────────────
    Rs_val = None
    if Rs_manual is not None and Rs_manual > 0:
        Rs_val = Rs_manual
        results["Rs"] = PVTResult("Solution Gas-Oil Ratio", "Rs", Rs_val, "scf/STB", "User Input")
    elif T is not None and api is not None and Pr is not None:
        if pvt_method == "Standing":
            Rs_val, w = calculate_rs_standing(Pr, T, api, gas_sg, Pb_val)
            src = "Standing Correlation"
        else:
            Rs_val, w = calculate_rs_vasquez_beggs(Pr, T, api, gas_sg, Psep, Tsep, Pb_val)
            src = "Vasquez-Beggs Correlation"
        all_warnings.extend(w)
        results["Rs"] = PVTResult("Solution Gas-Oil Ratio", "Rs", Rs_val, "scf/STB", src, w)
    else:
        msg = "Cannot compute Rs: need Pr, T, API."
        all_warnings.append(msg)
        results["Rs"] = PVTResult("Solution Gas-Oil Ratio", "Rs", None, "scf/STB", "N/A", [msg])

    # ── Step 3: Oil FVF (Bo) ──────────────────────────────────────────────
    Bo_val = None
    if Bo_manual is not None and Bo_manual > 0:
        Bo_val = Bo_manual
        results["Bo"] = PVTResult("Oil Formation Volume Factor", "Bo", Bo_val, "RB/STB", "User Input")
    elif Rs_val is not None and T is not None and api is not None:
        # Rs to use for Bo: at Pb if P > Pb, else at P
        Rs_for_Bo = Rs_val
        if pvt_method == "Standing":
            Bo_val, w = calculate_bo_standing(Rs_for_Bo, T, api, gas_sg)
            src = "Standing Correlation"
        else:
            Bo_val, w = calculate_bo_vasquez_beggs(Rs_for_Bo, T, api, gas_sg, Psep, Tsep)
            src = "Vasquez-Beggs Correlation"
        all_warnings.extend(w)

        # Undersaturated correction
        if Pb_val is not None and Pr is not None and Pr > Pb_val:
            # Recalculate Bo at Pb first
            if pvt_method == "Standing":
                Rs_at_Pb, _ = calculate_rs_standing(Pb_val, T, api, gas_sg)
                Bo_at_Pb, _ = calculate_bo_standing(Rs_at_Pb, T, api, gas_sg)
            else:
                Rs_at_Pb, _ = calculate_rs_vasquez_beggs(Pb_val, T, api, gas_sg, Psep, Tsep)
                Bo_at_Pb, _ = calculate_bo_vasquez_beggs(Rs_at_Pb, T, api, gas_sg, Psep, Tsep)

            Bo_val, w2 = calculate_bo_above_pb(
                Bo_at_Pb=Bo_at_Pb,
                P=Pr,
                Pb=Pb_val,
                Rs_at_Pb=Rs_at_Pb,
                T=T,
                api=api,
                gas_sg=gas_sg,
            )
            all_warnings.extend(w2)
            src += " + Undersaturated correction"

        results["Bo"] = PVTResult("Oil Formation Volume Factor", "Bo", Bo_val, "RB/STB", src, w)
    else:
        msg = "Cannot compute Bo: need Rs, T, API."
        all_warnings.append(msg)
        results["Bo"] = PVTResult("Oil Formation Volume Factor", "Bo", None, "RB/STB", "N/A", [msg])

    # ── Step 4: Oil Viscosity (μo) ────────────────────────────────────────

    mu_o_val = None

    if mu_o_manual is not None and mu_o_manual > 0:
        mu_o_val = mu_o_manual
        results["mu_o"] = PVTResult(
            "Oil Viscosity",
            "μo",
            mu_o_val,
            "cP",
            "User Input"
        )

    elif api is not None and T is not None and Pr is not None:

        # Step 4A: Calculate dead oil viscosity using Egbogah-type correlation

        mu_dead, w1 = calculate_dead_oil_viscosity_egbogah(api, T)
        all_warnings.extend(w1)

        # Case 1: Saturated condition, P <= Pb
        if Pb_val is not None and Pr <= Pb_val:
            # Rs at current pressure P = Pr
            if Rs_val is not None:
                Rs_current = Rs_val
            else:
                if pvt_method == "Standing":
                    Rs_current, w_rs = calculate_rs_standing(Pr, T, api, gas_sg, Pb_val)
                else:
                    Rs_current, w_rs = calculate_rs_vasquez_beggs(
                        Pr, T, api, gas_sg, Psep, Tsep, Pb_val
                    )
                all_warnings.extend(w_rs)

            mu_o_val, w2 = calculate_saturated_oil_viscosity_beggs_robinson(
                mu_dead=mu_dead,
                Rs=Rs_current,
            )
            all_warnings.extend(w2)

            src = "Egbogah Dead Oil + Beggs-Robinson Saturated Oil"

            results["mu_o"] = PVTResult(
                "Oil Viscosity",
                "μo",
                mu_o_val,
                "cP",
                src,
                w1 + w2,
            )

        # Case 2: Undersaturated condition, P > Pb
        elif Pb_val is not None and Pr > Pb_val:
            # Calculate Rs at bubble point first
            if pvt_method == "Standing":
                Rs_at_Pb, w_rs_pb = calculate_rs_standing(Pb_val, T, api, gas_sg)
            else:
                Rs_at_Pb, w_rs_pb = calculate_rs_vasquez_beggs(
                    Pb_val, T, api, gas_sg, Psep, Tsep
                )
            all_warnings.extend(w_rs_pb)

            # Calculate bubble-point oil viscosity μob using Beggs-Robinson
            mu_ob, w2 = calculate_saturated_oil_viscosity_beggs_robinson(
                mu_dead=mu_dead,
                Rs=Rs_at_Pb,
            )
            all_warnings.extend(w2)

            # Calculate undersaturated oil viscosity using Vasquez-Beggs
            mu_o_val, w3 = calculate_undersaturated_oil_viscosity(
                mu_ob=mu_ob,
                P=Pr,
                Pb=Pb_val,
            )
            all_warnings.extend(w3)

            src = (
                "Egbogah Dead Oil + Beggs-Robinson at Pb "
                "+ Vasquez-Beggs Undersaturated Correction"
            )

            results["mu_o"] = PVTResult(
                "Oil Viscosity",
                "μo",
                mu_o_val,
                "cP",
                src,
                w1 + w_rs_pb + w2 + w3,
            )

        # Case 3: Pb is unavailable, fallback to saturated viscosity at current Rs
        else:
            if Rs_val is not None:
                Rs_current = Rs_val
            elif GOR is not None:
                Rs_current = GOR
                all_warnings.append(
                    "Pb unavailable for viscosity calculation. "
                    "Using GOR as Rs fallback for saturated oil viscosity."
                )
            else:
                Rs_current = 0.0
                all_warnings.append(
                    "Pb and Rs unavailable for viscosity calculation. "
                    "Using Rs = 0 for saturated oil viscosity fallback."
                )

            mu_o_val, w2 = calculate_saturated_oil_viscosity_beggs_robinson(
                mu_dead=mu_dead,
                Rs=Rs_current,
            )
            all_warnings.extend(w2)

            src = "Egbogah Dead Oil + Beggs-Robinson Saturated Oil Fallback"

            results["mu_o"] = PVTResult(
                "Oil Viscosity",
                "μo",
                mu_o_val,
                "cP",
                src,
                w1 + w2,
            )

    else:
        msg = "Cannot compute μo: need API, T, and Pr."
        all_warnings.append(msg)
        results["mu_o"] = PVTResult(
            "Oil Viscosity",
            "μo",
            None,
            "cP",
            "N/A",
            [msg],
        )

    # ── Step 5: Gas Z-Factor ──────────────────────────────────────────────
    Z_val = None
    if Pr is not None and T is not None:
        Z_val, w = calculate_z_factor_dak(Pr, T, gas_sg)
        all_warnings.extend(w)
        results["Z"] = PVTResult("Gas Z-Factor", "Z", Z_val, "—", "DAK Correlation", w)
    else:
        results["Z"] = PVTResult("Gas Z-Factor", "Z", None, "—", "N/A",
                                 ["Cannot compute Z: need Pr, T."])

    # ── Step 6: Gas FVF (Bg) ──────────────────────────────────────────────
    Bg_val = None
    if Z_val is not None and Pr is not None and T is not None:
        Bg_val, w = calculate_bg(Pr, T, Z_val)
        all_warnings.extend(w)
        results["Bg"] = PVTResult("Gas Formation Volume Factor", "Bg", Bg_val, "RB/scf",
                                  "Calculated", w)
    else:
        results["Bg"] = PVTResult("Gas Formation Volume Factor", "Bg", None, "RB/scf", "N/A",
                                  ["Cannot compute Bg."])

    # ── Step 7: Gas Viscosity (μg) ────────────────────────────────────────
    mu_g_val = None
    if Z_val is not None and Pr is not None and T is not None:
        mu_g_val, w = calculate_gas_viscosity_lge(Pr, T, gas_sg, Z_val)
        all_warnings.extend(w)
        results["mu_g"] = PVTResult("Gas Viscosity", "μg", mu_g_val, "cP",
                                    "Lee-Gonzalez-Eakin Correlation", w)
    else:
        results["mu_g"] = PVTResult("Gas Viscosity", "μg", None, "cP", "N/A",
                                    ["Cannot compute μg."])

    # ── Step 8: Water Properties (optional) ───────────────────────────────
    if Pr is not None and T is not None:
        Bw_val, w1 = calculate_bw(Pr, T)
        mu_w_val, w2 = calculate_muw(Pr, T, salinity)
        results["Bw"] = PVTResult("Water FVF", "Bw", Bw_val, "RB/STB", "McCain Correlation", w1)
        results["mu_w"] = PVTResult("Water Viscosity", "μw", mu_w_val, "cP", "McCain Correlation", w2)
        all_warnings.extend(w1 + w2)

    # ── Build values dict ─────────────────────────────────────────────────
    for key, res in results.items():
        values[key] = res.value

    return {
        "results": results,
        "warnings": all_warnings,
        "values": values,
    }
