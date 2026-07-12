from __future__ import annotations

import math
from typing import Dict, Tuple, Any


#  CONSTANTS

G_C = 32.174          # lbm·ft / (lbf·s²)
EPS = 1.0e-12         # small number to avoid division-by-zero


# ══════════════════════════════════════════════════════════════════════════════
#  BASIC GUARDS / HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(value: float, low: float, high: float) -> float:
    """Limit a value between low and high."""
    return max(low, min(value, high))


def _safe_log(x: float) -> float:
    """Natural logarithm with a positive lower guard."""
    return math.log(max(x, 1.0e-30))


def _smooth_step(x: float, delta: float = 0.3) -> float:
    """
    Smooth Heaviside step function (Teixeira et al., 2015).

    g(x; δ) = (1 + tanh(x / δ)) / 2

    Returns ≈ 1 when x ≫ δ, ≈ 0 when x ≪ −δ, exactly 0.5 when x = 0.

    Reference
    ---------
    Teixeira, Secchi & Biscaia (2015), "Two-Phase Flow in Pipes: Numerical
    Improvements and Qualitative Analysis," Oil & Gas Science and Technology
    (IFP Energies Nouvelles), DOI: 10.2516/ogst/2013191.
    """
    arg = x / max(abs(delta), 1.0e-12)
    arg = _clamp(arg, -20.0, 20.0)  # prevent tanh overflow
    return 0.5 * (1.0 + math.tanh(arg))


def _griffith_holdup_bb(mixture_velocity: float, superficial_gas_velocity: float, no_slip_liquid_fraction: float) -> float:
    """
    Griffith (1961) Bubble Flow Liquid Holdup
    """
    bubble_slip_velocity = 0.8
    discriminant = (1.0 + mixture_velocity / bubble_slip_velocity)**2 - 4.0 * superficial_gas_velocity / bubble_slip_velocity
    discriminant = max(discriminant, 0.0)
    
    liquid_holdup = 1.0 - 0.5 * (1.0 + mixture_velocity / bubble_slip_velocity - math.sqrt(discriminant))
    
    return max(min(liquid_holdup, 1.0), no_slip_liquid_fraction)


def _resolve_inclination(
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
) -> tuple[float, str]:
    """
    Convert user input into a signed inclination angle.

    Beggs-Brill uses inclination angle from horizontal:
        +90° = vertical upward / uphill production
          0° = horizontal
        -90° = vertical downward / downhill flow

    In the UI, we want to keep it simple:
        Flow Direction = Uphill / Production  → positive angle
        Flow Direction = Downhill / Injection → negative angle

    Parameters
    ----------
    inclination_deg : float
        Magnitude of inclination angle from horizontal, degrees.
        The code uses abs(inclination_deg), then applies sign based on flow_direction.
    flow_direction : str
        Any string containing 'down' or 'inject' is treated as downhill.
        Any string containing 'up' or 'prod' is treated as uphill.

    Returns
    -------
    tuple[float, str]
        signed inclination angle in degrees, normalized direction label
    """
    # Keep physical inclination magnitude between 0 and 90 degrees.
    angle = _clamp(abs(float(inclination_deg)), 0.0, 90.0)

    direction = (flow_direction or "Uphill / Production").strip().lower()

    if "down" in direction or "inject" in direction:
        return -angle, "Downhill / Injection"

    # Default for production nodal analysis.
    return angle, "Uphill / Production"


def inclination_from_md_tvd(
    MD: float,
    TVD: float,
    flow_direction: str = "Uphill / Production",
) -> float:
    """
    Utility helper to estimate average inclination from MD and TVD.

    This is optional. You can use this in vlp.py when only MD and TVD are known.

    For an average wellbore section:
        sin(alpha) = TVD / MD

    where:
        alpha = inclination angle from horizontal
        MD    = measured pipe length, ft
        TVD   = true vertical depth, ft

    Returns
    -------
    float
        Signed average inclination angle, degrees.
    """
    MD = max(float(MD), EPS)
    TVD = _clamp(float(TVD), 0.0, MD)

    angle = math.degrees(math.asin(_clamp(TVD / MD, 0.0, 1.0)))
    signed_angle, _ = _resolve_inclination(angle, flow_direction)
    return signed_angle


# ══════════════════════════════════════════════════════════════════════════════
#  BEGGS-BRILL DIMENSIONLESS NUMBERS
# ══════════════════════════════════════════════════════════════════════════════

def _froude_number(v_m: float, d_in: float) -> float:
    """
    Beggs-Brill Froude number using diameter in inches.

        Nfr = 12 Vm² / (g d) ≈ 0.373 Vm² / d

    Field-unit form used here:
        Vm = ft/s
        d  = inches
    """
    d_in = max(d_in, EPS)
    return 0.373 * (v_m ** 2) / d_in


def _liquid_velocity_number(v_sl: float, rho_l: float, sigma_l: float) -> float:
    """
    Beggs-Brill liquid velocity number Nlv.

    Field units:
        Vsl     = ft/s
        rho_l   = lb/ft³
        sigma_l = dynes/cm
    """
    sigma_l = max(sigma_l, 0.1)
    rho_l = max(rho_l, EPS)
    return 1.938 * v_sl * (rho_l / sigma_l) ** 0.25


# ══════════════════════════════════════════════════════════════════════════════
#  FLOW-PATTERN BOUNDARIES AND FLOW-PATTERN SELECTION
# ══════════════════════════════════════════════════════════════════════════════

def _flow_pattern_boundaries(lambda_l: float) -> tuple[float, float, float, float]:
    """
    Calculate Beggs-Brill flow-pattern boundary parameters L1, L2, L3, L4.
    """
    lambda_l = max(lambda_l, 1.0e-8)

    L1 = 316.0    * lambda_l ** 0.302
    L2 = 0.0009252 * lambda_l ** (-2.4684)
    L3 = 0.10     * lambda_l ** (-1.4516)
    L4 = 0.5      * lambda_l ** (-6.738)

    return L1, L2, L3, L4


def _flow_pattern(lambda_l: float, Nfr: float) -> tuple[str, float, float, float, float]:
    """
    Select Beggs-Brill horizontal flow pattern using the user's confirmed logic.

    Flow patterns:
        - segregated
        - transition
        - intermittent
        - distributed

    """
    lambda_l = max(lambda_l, 1.0e-8)
    L1, L2, L3, L4 = _flow_pattern_boundaries(lambda_l)

    # 1. Segregated flow
    if (lambda_l < 0.01 and Nfr < L1) or (lambda_l >= 0.01 and Nfr < L2):
        return "segregated", L1, L2, L3, L4

    # 2. Transition flow
    if lambda_l >= 0.01 and L2 <= Nfr <= L3:
        return "transition", L1, L2, L3, L4

    # 3. Intermittent flow
    if (0.01 <= lambda_l < 0.4 and L3 < Nfr <= L1) or (lambda_l >= 0.4 and L3 < Nfr <= L4):
        return "intermittent", L1, L2, L3, L4

    # 4. Distributed flow
    if (lambda_l < 0.4 and Nfr >=L1) or (lambda_l >= 0.4 and Nfr > L4):
        return "distributed", L1, L2, L3, L4


def _smooth_regime_weights(
    lambda_l: float,
    Nfr: float,
    L1: float,
    L2: float,
    L3: float,
    L4: float,
    delta: float = 0.3,
) -> tuple[float, float, float, float]:
    """
    Compute smooth, continuous regime weights via Teixeira regularization.

    Instead of selecting one flow regime via hard if/elif boundaries,
    every regime receives a continuous weight in [0, 1].  The weights are
    normalised to sum to exactly 1.

    Log-space arguments are used — g(ln(NFR / L); δ) rather than
    g(NFR − L; δ) — so that the blend width is proportional to each
    boundary's own magnitude.  This is critical when L-values span many
    orders of magnitude (e.g. L2 ≈ 0.1 vs L4 ≈ 200 000).

    Reference
    ---------
    Teixeira, Secchi & Biscaia (2015), DOI: 10.2516/ogst/2013191.

    Parameters
    ----------
    lambda_l : float   No-slip liquid fraction, 0 < λL ≤ 1.
    Nfr      : float   Froude number.
    L1–L4    : float   Beggs-Brill flow-pattern boundary values.
    delta    : float   Blend half-width in log-space (default 0.3 ≈ ±35 %).

    Returns
    -------
    (w_seg, w_tran, w_int, w_dist) — regime weights, sum = 1.
    """
    # Guard all inputs to positive values for safe log()
    lambda_l = max(lambda_l, 1.0e-8)
    Nfr = max(Nfr, 1.0e-10)
    L1 = max(L1, 1.0e-10)
    L2 = max(L2, 1.0e-10)
    L3 = max(L3, 1.0e-10)
    L4 = max(L4, 1.0e-10)

    # ── Smooth indicators for λL thresholds ────────────────────────────────
    # "below 0.01" → ≈ 1 when λL ≪ 0.01
    lam_below_001 = _smooth_step(_safe_log(0.01 / lambda_l), delta)
    lam_above_001 = 1.0 - lam_below_001

    # "below 0.4"  → ≈ 1 when λL ≪ 0.4
    lam_below_04 = _smooth_step(_safe_log(0.4 / lambda_l), delta)
    lam_above_04 = 1.0 - lam_below_04

    # ── Smooth indicators for NFR vs L boundaries ─────────────────────────
    nfr_below_L1 = _smooth_step(_safe_log(L1 / Nfr), delta)
    nfr_above_L1 = 1.0 - nfr_below_L1

    nfr_below_L2 = _smooth_step(_safe_log(L2 / Nfr), delta)
    nfr_above_L2 = 1.0 - nfr_below_L2

    nfr_below_L3 = _smooth_step(_safe_log(L3 / Nfr), delta)
    nfr_above_L3 = 1.0 - nfr_below_L3

    nfr_below_L4 = _smooth_step(_safe_log(L4 / Nfr), delta)
    nfr_above_L4 = 1.0 - nfr_below_L4

    # ── Regime indicators (mirror the hard B&B map structure) ──────────────
    #
    # Segregated:   (λL < 0.01 ∧ NFR < L1)  ∨  (λL ≥ 0.01 ∧ NFR < L2)
    w_seg = lam_below_001 * nfr_below_L1 + lam_above_001 * nfr_below_L2

    # Transition:   λL ≥ 0.01  ∧  L2 ≤ NFR ≤ L3
    w_tran = lam_above_001 * nfr_above_L2 * nfr_below_L3

    # Intermittent: (0.01 ≤ λL < 0.4 ∧ L3 < NFR ≤ L1)
    #             ∨ (λL ≥ 0.4         ∧ L3 < NFR ≤ L4)
    w_int = (lam_above_001 * lam_below_04 * nfr_above_L3 * nfr_below_L1
             + lam_above_04 * nfr_above_L3 * nfr_below_L4)

    # Distributed:  (λL < 0.4 ∧ NFR ≥ L1)  ∨  (λL ≥ 0.4 ∧ NFR > L4)
    w_dist = lam_below_04 * nfr_above_L1 + lam_above_04 * nfr_above_L4

    # ── Normalize so weights sum to exactly 1 ─────────────────────────────
    total = w_seg + w_tran + w_int + w_dist
    if total > 1.0e-12:
        inv = 1.0 / total
        w_seg *= inv
        w_tran *= inv
        w_int *= inv
        w_dist *= inv
    else:
        # Extremely unlikely fallback — default to segregated
        w_seg, w_tran, w_int, w_dist = 1.0, 0.0, 0.0, 0.0

    return w_seg, w_tran, w_int, w_dist


# ══════════════════════════════════════════════════════════════════════════════
#  HORIZONTAL LIQUID HOLDUP HL(0)
# ══════════════════════════════════════════════════════════════════════════════

# Coefficients for:
#     HL(0) = a · lambda_l^b / Nfr^c
#
# These values are used for horizontal holdup before inclination correction.
_HL0_COEFFS = {
    "segregated":   (0.980, 0.4846, 0.0868),
    "intermittent": (0.845, 0.5351, 0.0173),
    "distributed":  (1.065, 0.5824, 0.0609),
}


def _horizontal_holdup(pattern: str, lambda_l: float, Nfr: float) -> float:
    """
    Calculate horizontal liquid holdup HL(0).

    For normal regimes:
        HL(0) = a · lambda_l^b / Nfr^c

    """
    if pattern not in _HL0_COEFFS:
        raise ValueError(f"Unsupported holdup pattern for HL(0): {pattern}")

    a, b, c = _HL0_COEFFS[pattern]
    lambda_l = max(lambda_l, 1.0e-8)
    Nfr = max(Nfr, 1.0e-10)

    HL0 = a * (lambda_l ** b) / (Nfr ** c)

    # Physical guard: horizontal holdup cannot be less than no-slip liquid fraction.
    return _clamp(HL0, lambda_l, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  INCLINATION CORRECTION ψ
# ══════════════════════════════════════════════════════════════════════════════

# Coefficients for:
#     C = (1 - lambda_l) · ln(e · lambda_l^f · Nlv^g · Nfr^h)

_UPHILL_COEFFS = {
    "segregated":   (0.011, -3.7680,  3.5390, -1.6140),
    "intermittent": (2.960,  0.3050, -0.4473,  0.0978),
}

_DOWNHILL_COEFFS = (4.700, -0.3692, 0.1244, -0.5056)


def _inclination_correction(
    pattern: str,
    lambda_l: float,
    Nfr: float,
    Nlv: float,
    alpha_deg: float,
) -> tuple[float, float]:
    """
    Calculate Beggs-Brill inclination correction factor ψ.

    Formula structure:
        HL(alpha) = HL(0) · ψ

        ψ = 1 + C [sin(1.8α) - (1/3) sin³(1.8α)]

    where:
        α is inclination angle from horizontal in degrees.
        α > 0 → uphill
        α < 0 → downhill

    Constraint:
        C >= 0

    Returns
    -------
    tuple[float, float]
        psi, C
    """
    lambda_l = max(lambda_l, 1.0e-8)
    Nfr = max(Nfr, 1.0e-10)
    Nlv = max(Nlv, 1.0e-10)

    # Horizontal pipe: no inclination correction.
    if abs(alpha_deg) < 1.0e-12:
        return 1.0, 0.0

    # For transition flow, B&B uses intermittent coefficients for inclination
    # when a single effective pattern is needed. In this module, true transition
    # holdup is calculated by separately blending segregated and intermittent.
    # This fallback is only for direct calls with pattern='transition'.
    effective_pattern = "intermittent" if pattern == "transition" else pattern

    # Select coefficient set.
    if alpha_deg > 0.0:
        # Uphill distributed flow has no correction.
        if effective_pattern == "distributed":
            return 1.0, 0.0

        e, f, g, h = _UPHILL_COEFFS.get(effective_pattern, _UPHILL_COEFFS["intermittent"])
    else:
        # Downhill uses one coefficient set for all patterns.
        e, f, g, h = _DOWNHILL_COEFFS

    arg = e * (lambda_l ** f) * (Nlv ** g) * (Nfr ** h)
    C = (1.0 - lambda_l) * _safe_log(arg)

    # User-confirmed constraint: C must be non-negative.
    C = max(C, 0.0)

    # Important: sin() expects radians.
    # Using sin(radians(1.8 * alpha_deg)) is equivalent to sin(1.8α)
    # when alpha is supplied in degrees.
    sin_term = math.sin(math.radians(1.8 * alpha_deg))
    shape = sin_term - (1.0 / 3.0) * (sin_term ** 3)

    psi = 1.0 + C * shape

    # Downhill correction can reduce psi. Do not force psi >= 1.
    # Only protect against non-physical negative/zero values.
    psi = max(psi, 1.0e-6)

    return psi, C


# ══════════════════════════════════════════════════════════════════════════════
#  LIQUID HOLDUP INCLUDING TRANSITION AND PAYNE CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def _holdup_for_non_transition_pattern(
    pattern: str,
    lambda_l: float,
    Nfr: float,
    Nlv: float,
    alpha_deg: float,
) -> dict[str, float]:
    """
    Calculate inclined liquid holdup for a non-transition pattern.

    This function returns intermediate values also, so the detailed VLP table
    can show HL(0), psi, C, and HL before Payne correction.
    """
    HL0 = _horizontal_holdup(pattern, lambda_l, Nfr)
    psi, C = _inclination_correction(pattern, lambda_l, Nfr, Nlv, alpha_deg)
    HL_alpha = _clamp(HL0 * psi, lambda_l, 1.0)

    return {
        "HL0": HL0,
        "psi": psi,
        "C": C,
        "HL_alpha": HL_alpha,
    }


def _transition_blend_factor(lambda_l: float, Nfr: float, L2: float, L3: float) -> float:
    """
    Calculate transition blending factor B.

    User formula:
        B = [0.1 lambda_l^-1.452 - Nfr]
            / [0.1 lambda_l^-1.452 - 0.000925 lambda_l^-2.4684]

    Since:
        L3 = 0.10 lambda_l^-1.4516
        L2 = 0.0009252 lambda_l^-2.4684

    this is equivalent to:
        B = (L3 - Nfr) / (L3 - L2)

    For safety, B is clamped between 0 and 1.
    """
    denom = max(L3 - L2, 1.0e-10)
    B = (L3 - Nfr) / denom
    return _clamp(B, 0.0, 1.0)


def _beggs_brill_holdup(
    pattern: str,
    lambda_l: float,
    Nfr: float,
    Nlv: float,
    alpha_deg: float,
    L1: float,
    L2: float,
    L3: float,
    L4: float,
    apply_payne: bool = True,
) -> dict[str, float | str]:
    """
    Calculate final Beggs-Brill liquid holdup with smooth regime blending.

    Uses the Teixeira regularization (DOI: 10.2516/ogst/2013191) to replace
    hard regime-boundary switches with continuous sigmoid weights.  This
    eliminates the unphysical pressure-gradient discontinuities that occur
    when NFR crosses a boundary (e.g. Intermittent ↔ Distributed at L1).

    Sequence:
        1. Compute smooth regime weights (Teixeira regularization)
        2. Compute inclined holdup for segregated, intermittent, distributed
        3. Fold transition weight into segregated / intermittent via B
        4. Blend holdups using effective weights
        5. Apply Payne correction
        6. Apply physical guard: lambda_l <= HL <= 1

    Far from any boundary, exactly one weight ≈ 1 and the rest ≈ 0, so
    results are identical to standard Beggs-Brill.
    """
    # ── 1. Smooth regime weights ───────────────────────────────────────────
    w_seg, w_tran, w_int, w_dist = _smooth_regime_weights(
        lambda_l, Nfr, L1, L2, L3, L4,
    )

    # ── 2. Compute inclined holdup for each base regime ────────────────────
    seg = _holdup_for_non_transition_pattern(
        "segregated", lambda_l, Nfr, Nlv, alpha_deg,
    )
    inter = _holdup_for_non_transition_pattern(
        "intermittent", lambda_l, Nfr, Nlv, alpha_deg,
    )
    dist = _holdup_for_non_transition_pattern(
        "distributed", lambda_l, Nfr, Nlv, alpha_deg,
    )

    # ── 3. Transition blend factor B ───────────────────────────────────────
    # Standard B&B:  HL_transition = B · HL_seg + (1−B) · HL_int
    # Fold transition weight into segregated and intermittent.
    B = _transition_blend_factor(lambda_l, Nfr, L2, L3)

    eff_seg = w_seg + w_tran * B
    eff_int = w_int + w_tran * (1.0 - B)
    eff_dist = w_dist

    # ── 4. Blended inclined holdup ────────────────────────────────────────
    HL_alpha = (eff_seg * seg["HL_alpha"]
                + eff_int * inter["HL_alpha"]
                + eff_dist * dist["HL_alpha"])

    # Weighted diagnostic values for table / debugging.
    HL0 = (eff_seg * seg["HL0"]
           + eff_int * inter["HL0"]
           + eff_dist * dist["HL0"])
    psi = (eff_seg * seg["psi"]
           + eff_int * inter["psi"]
           + eff_dist * dist["psi"])
    C = (eff_seg * seg["C"]
         + eff_int * inter["C"]
         + eff_dist * dist["C"])

    # Report transition B only when transition weight is significant.
    transition_B = B if w_tran > 0.01 else None

    # ── 5. Payne correction ───────────────────────────────────────────────
    # User-confirmed default:
    #   Uphill   → HL_corrected = 0.924 * HL(alpha)
    #   Downhill → HL_corrected = 0.685 * HL(alpha)
    if not apply_payne or abs(alpha_deg) < 1.0e-12:
        payne_factor = 1.0
    elif alpha_deg > 0.0:
        payne_factor = 0.924
    else:
        payne_factor = 0.685

    HL_payne = payne_factor * HL_alpha

    # ── 6. Physical guard ─────────────────────────────────────────────────
    # Payne correction can reduce HL below lambda_l, so this guard matters.
    HL_final = _clamp(HL_payne, lambda_l, 1.0)

    return {
        "pattern": pattern,
        "HL0": HL0,
        "psi": psi,
        "C": C,
        "HL_alpha_before_payne": HL_alpha,
        "payne_factor": payne_factor,
        "H_L": HL_final,
        "transition_B": transition_B,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FRICTION FACTOR CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _friction_churchill_smooth(Re: float, d_in: float) -> float:

    Re = max(Re, EPS)
    if Re <= 0.0:
        return 0.025

    # User requested roughness = 0 for Beggs-Brill friction calculation.
    e_d = 0.0

    A = (-2.457 * math.log((7.0 / Re) ** 0.9 + 0.27 * e_d)) ** 16
    B = (37530.0 / Re) ** 16

    # Darcy friction factor.
    f_D = 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)
    return max(f_D, 1.0e-8)


def _friction_smooth_beggs_brill(Re: float) -> float:
    """
    Smooth-pipe friction factor option requested by the user.

    Darcy friction factor:
        Laminar:      f = 64 / Re, for Re <= 2100
        Transition:   linear interpolation, for 2100 < Re <= 4000
        Turbulent:    f = [2 log10(Re / (4.5223 log10(Re) - 3.8215))]^-2

    Important:
        This is Darcy friction factor, not Fanning.
    """
    Re = max(Re, EPS)

    def _turbulent_smooth(re_value: float) -> float:
        re_value = max(re_value, 4000.0)
        log_re = math.log10(re_value)
        denom_inside = 4.5223 * log_re - 3.8215
        denom_inside = max(denom_inside, 1.0e-12)
        term = 2.0 * math.log10(re_value / denom_inside)
        return 1.0 / max(term ** 2, 1.0e-12)

    if Re <= 2100.0:
        return 64.0 / Re

    if Re <= 4000.0:
        f_lam_2100 = 64.0 / 2100.0
        f_turb_4000 = _turbulent_smooth(4000.0)
        weight = (Re - 2100.0) / (4000.0 - 2100.0)
        return f_lam_2100 + weight * (f_turb_4000 - f_lam_2100)

    return _turbulent_smooth(Re)


def _base_friction_factor(Re: float, d_in: float, friction_method: str = "Churchill") -> tuple[float, str]:
    """
    Select the base no-slip Darcy friction factor.

    Supported method strings:
        - 'Churchill'
        - 'Beggs-Brill Smooth'
        - 'Colebrook Smooth'
        - 'Smooth Pipe'

    Roughness is intentionally not used because the user confirmed
    roughness = 0 for the Beggs-Brill friction step.
    """
    method = (friction_method or "Churchill").strip().lower()

    if "smooth" in method or "colebrook" in method or "colbroke" in method:
        return _friction_smooth_beggs_brill(Re), "Beggs-Brill Smooth Pipe"

    return _friction_churchill_smooth(Re, d_in), "Churchill Smooth Pipe"


# ══════════════════════════════════════════════════════════════════════════════
#  TWO-PHASE FRICTION MULTIPLIER
# ══════════════════════════════════════════════════════════════════════════════

def _two_phase_friction_multiplier(lambda_l: float, H_L: float) -> tuple[float, float, float]:
    """
    Calculate Beggs-Brill two-phase friction multiplier.

    y = lambda_l / H_L²

    If:
        1 < y < 1.2

    use special branch:
        S = ln(2.2y - 1.2)

    Otherwise:
        S = ln(y) / [-0.0523 + 3.182ln(y) - 0.8725ln(y)² + 0.01852ln(y)^4]

    Then:
        ftp = fn · exp(S)

    Returns
    -------
    tuple[float, float, float]
        y, S, exp(S)
    """
    lambda_l = max(lambda_l, 1.0e-8)
    H_L = max(H_L, 1.0e-8)

    y = lambda_l / (H_L ** 2)
    y = max(y, 1.0e-10)

    if 1.0 < y < 1.2:
        S = math.log(2.2 * y - 1.2)
    else:
        ln_y = math.log(y)
        denominator = (
            -0.0523
            + 3.182 * ln_y
            - 0.8725 * (ln_y ** 2)
            + 0.01852 * (ln_y ** 4)
        )

        if abs(denominator) > 1.0e-10:
            S = ln_y / denominator
        else:
            S = 0.0

    multiplier = math.exp(S)
    return y, S, multiplier


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC UTILITY FUNCTIONS FOR TESTING / TABLES
# ══════════════════════════════════════════════════════════════════════════════

def flow_regime_beggs_brill(fp: dict) -> str:
    """Return only the Beggs-Brill flow-regime label for a fluid-property state."""
    v_m = max(float(fp.get("v_m", 0.0)), 0.0)
    d_in = max(float(fp.get("d_in", 0.0)), EPS)
    lambda_l = _clamp(float(fp.get("lam_l", 1.0)), 1.0e-8, 1.0)

    if v_m < 1.0e-8:
        return "segregated"

    Nfr = _froude_number(v_m, d_in)
    pattern, _, _, _, _ = _flow_pattern(lambda_l, Nfr)
    return pattern


def liquid_holdup_beggs_brill(
    fp: dict,
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
    apply_payne: bool = True,
) -> float:
    """Return only final Beggs-Brill liquid holdup for a fluid-property state."""
    _, details = pressure_gradient_beggs_brill(
        fp=fp,
        inclination_deg=inclination_deg,
        flow_direction=flow_direction,
        friction_method="Churchill",
        apply_payne=apply_payne,
    )
    return float(details["H_L"])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def pressure_gradient_beggs_brill(
    fp: Dict[str, float],
    inclination_deg: float = 90.0,
    flow_direction: str = "Uphill / Production",
    friction_method: str = "Churchill",
    apply_payne: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate Beggs-Brill total pressure gradient for one pipe segment.

    Parameters
    ----------
    fp : dict
        Fluid-property dictionary from vlp.py.
        Required keys:
            p, d_in, rho_l, rho_g, mu_l, mu_g, sigma,
            v_sl, v_sg, v_m, lam_l

    inclination_deg : float, default 90.0
        Inclination magnitude from horizontal, degrees.
        90 = vertical, 0 = horizontal.

    flow_direction : str, default 'Uphill / Production'
        Controls sign of inclination and Payne correction:
            Uphill / Production   → positive inclination
            Downhill / Injection  → negative inclination

    friction_method : str, default 'Churchill'
        Base no-slip friction factor option:
            - Churchill
            - Beggs-Brill Smooth / Colebrook Smooth / Smooth Pipe

    apply_payne : bool, default True
        Apply Payne correction after inclined holdup calculation:
            Uphill   → 0.924 × HL(alpha)
            Downhill → 0.685 × HL(alpha)

    Returns
    -------
    tuple[float, dict]
        total_gradient, details

        total_gradient is in psi/ft of measured pipe length.
        In vlp.py, for Beggs-Brill use:
            dP_segment = total_gradient × dL
        where:
            dL = MD / n_segments
    """

    # ── 1. Unpack fluid properties from vlp.py fp dictionary ───────────────
    pressure_psia = max(float(fp["p"]), 1.0)
    pipe_id_inches = max(float(fp["d_in"]), EPS)

    liquid_density = max(float(fp["rho_l"]), EPS)
    gas_density = max(float(fp["rho_g"]), EPS)

    liquid_viscosity = max(float(fp["mu_l"]), EPS)
    gas_viscosity = max(float(fp["mu_g"]), 1.0e-8)

    surface_tension = max(float(fp["sigma"]), 0.1)

    superficial_liquid_velocity = max(float(fp["v_sl"]), 0.0)
    superficial_gas_velocity = max(float(fp["v_sg"]), 0.0)
    mixture_velocity = max(float(fp["v_m"]), 0.0)

    # Use fp['lam_l'] if available, but recompute safely if needed.
    if mixture_velocity > 1.0e-10:
        lambda_l = superficial_liquid_velocity / mixture_velocity
    else:
        lambda_l = float(fp.get("lam_l", 1.0))

    lambda_l = _clamp(lambda_l, 1.0e-8, 1.0)
    lambda_g = 1.0 - lambda_l

    # Resolve signed inclination angle.
    alpha_deg, direction_label = _resolve_inclination(inclination_deg, flow_direction)
    sin_alpha = math.sin(math.radians(alpha_deg))

    # ── 2. Static / near-zero-flow guard ───────────────────────────────────
    # If Vm is almost zero, use static liquid column only.
    # This keeps the function safe if accidentally called at q = 0.
    if mixture_velocity < 1.0e-8:
        H_L = 1.0
        mixture_density = liquid_density
        elevational_gradient = mixture_density * sin_alpha / 144.0
        frictional_gradient = 0.0
        acceleration_term = 0.0
        total_gradient = elevational_gradient

        details = {
            "method": "Beggs-Brill",
            "flow_regime": "static",
            "flow_direction": direction_label,
            "inclination_deg": alpha_deg,
            "lambda_ns": lambda_l,
            "lambda_g": lambda_g,
            "Nfr": 0.0,
            "Nlv": 0.0,
            "L1": None,
            "L2": None,
            "L3": None,
            "L4": None,
            "HL0": H_L,
            "psi": 1.0,
            "C": 0.0,
            "transition_B": None,
            "HL_alpha_before_payne": H_L,
            "payne_factor": 1.0,
            "H_L": H_L,
            "rho_s": mixture_density,
            "rho_n": liquid_density,
            "mu_n": liquid_viscosity,
            "Re": 0.0,
            "friction_method": friction_method,
            "f_D": 0.0,
            "y": None,
            "S": 0.0,
            "friction_multiplier": 1.0,
            "f_tp": 0.0,
            "frictional_gradient": frictional_gradient,
            "elevational_gradient": elevational_gradient,
            "Ek": acceleration_term,
            "accel_gradient": 0.0,
            "total_gradient": total_gradient,
            "V_m": mixture_velocity,
            "V_sl": superficial_liquid_velocity,
            "V_sg": superficial_gas_velocity,
        }

        return total_gradient, details

    # ── 3. Calculate Beggs-Brill dimensionless numbers ─────────────────────
    Nfr = _froude_number(mixture_velocity, pipe_id_inches)
    Nlv = _liquid_velocity_number(superficial_liquid_velocity, liquid_density, surface_tension)

    # ── 4. Calculate L1, L2, L3, L4 and select flow pattern ────────────────
    pattern, L1, L2, L3, L4 = _flow_pattern(lambda_l, Nfr)

    # ── 5. Calculate liquid holdup HL ──────────────────────────────────────
    holdup_info = _beggs_brill_holdup(
        pattern=pattern,
        lambda_l=lambda_l,
        Nfr=Nfr,
        Nlv=Nlv,
        alpha_deg=alpha_deg,
        L1=L1,
        L2=L2,
        L3=L3,
        L4=L4,
        apply_payne=apply_payne,
    )

    liquid_holdup = float(holdup_info["H_L"])

    # ── 5b. Griffith Bubble Flow Correction for Uphill Flow ────────────────
    # Apply Griffith-Wallis bubble flow check for vertical/uphill flow
    # This prevents Beggs-Brill from overpredicting holdup in bubble flow
    if flow_direction == "Uphill / Production":
        L_B = 1.071 - (0.2662 * (mixture_velocity**2) / pipe_id_inches)
        L_B = max(L_B, 0.13)
        
        pure_griffith_threshold = 0.7 * L_B
        pure_bb_threshold = L_B
        
        if lambda_g < pure_griffith_threshold:
            liquid_holdup = _griffith_holdup_bb(mixture_velocity, superficial_gas_velocity, lambda_l)
            pattern = "griffith_bubble"
        elif lambda_g >= pure_bb_threshold:
            pass # Use standard Beggs-Brill holdup
        else:
            # Transition blend
            blend_factor = (lambda_g - pure_griffith_threshold) / max(pure_bb_threshold - pure_griffith_threshold, 1e-10)
            griffith_hl = _griffith_holdup_bb(mixture_velocity, superficial_gas_velocity, lambda_l)
            liquid_holdup = (1.0 - blend_factor) * griffith_hl + blend_factor * liquid_holdup
            pattern = "griffith_bb_blend"
            
        liquid_holdup = _clamp(liquid_holdup, lambda_l, 1.0)

    # ── 6. Calculate slip mixture density for elevation term ───────────────
    # This uses final Payne-corrected holdup.
    mixture_density = liquid_density * liquid_holdup + gas_density * (1.0 - liquid_holdup)

    # ── 7. Elevation gradient ──────────────────────────────────────────────
    # Output is psi/ft.
    # 144 converts lbf/ft² to psi.
    elevational_gradient = mixture_density * sin_alpha / 144.0

    # ── 8. No-slip mixture properties for friction/Reynolds number ─────────
    no_slip_density = liquid_density * lambda_l + gas_density * (1.0 - lambda_l)

    # Beggs-Brill uses no-slip linear-blend viscosity.
    no_slip_viscosity = liquid_viscosity * lambda_l + gas_viscosity * (1.0 - lambda_l)
    no_slip_viscosity = max(no_slip_viscosity, EPS)

    # Reynolds number with d in inches:
    #     Re = 124 ρns Vm d / μns
    reynolds_number = 124.0 * no_slip_density * mixture_velocity * pipe_id_inches / no_slip_viscosity

    # ── 9. Base no-slip Darcy friction factor ──────────────────────────────
    base_friction_factor, friction_method_used = _base_friction_factor(
        reynolds_number,
        pipe_id_inches,
        friction_method,
    )

    # ── 10. Two-phase friction multiplier ──────────────────────────────────
    y_param, S, friction_multiplier = _two_phase_friction_multiplier(lambda_l, liquid_holdup)
    two_phase_friction_factor = base_friction_factor * friction_multiplier

    # ── 11. Friction gradient ──────────────────────────────────────────────
    # With d in inches:
    #     grad_f = ftp ρns Vm² / (24 gc d)
    #
    # This is equivalent to:
    #     grad_f = ftp ρns Vm² / (2 gc D 144)
    # where D = d / 12 ft.
    frictional_gradient = (
        two_phase_friction_factor
        * no_slip_density
        * (mixture_velocity ** 2)
        / (24.0 * G_C * pipe_id_inches)
    )

    # ── 12. Acceleration / kinetic correction ──────────────────────────────
    # Ek = ρm Vm Vsg / (gc P 144)
    acceleration_term = (
        mixture_density
        * mixture_velocity
        * superficial_gas_velocity
        / (G_C * pressure_psia * 144.0)
    )

    # Guard against denominator collapse.
    acceleration_term = _clamp(acceleration_term, 0.0, 0.99)

    # ── 13. Total pressure gradient ────────────────────────────────────────
    total_gradient = (elevational_gradient + frictional_gradient) / (1.0 - acceleration_term)

    # Acceleration contribution shown in detailed table.
    accel_gradient = total_gradient * acceleration_term

    # ── 14. Details dictionary for debugging / detailed VLP table ──────────
    details = {
        "method": "Beggs-Brill",
        "flow_regime": pattern,
        "flow_direction": direction_label,
        "inclination_deg": alpha_deg,
        "lambda_ns": lambda_l,
        "lambda_g": lambda_g,
        "Nfr": Nfr,
        "Nlv": Nlv,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4,
        "HL0": holdup_info["HL0"],
        "psi": holdup_info["psi"],
        "C": holdup_info["C"],
        "transition_B": holdup_info["transition_B"],
        "HL_alpha_before_payne": holdup_info["HL_alpha_before_payne"],
        "payne_factor": holdup_info["payne_factor"],
        "H_L": liquid_holdup,
        "rho_s": mixture_density,
        "rho_n": no_slip_density,
        "mu_n": no_slip_viscosity,
        "Re": reynolds_number,
        "friction_method": friction_method_used,
        "f_D": base_friction_factor,
        "y": y_param,
        "S": S,
        "friction_multiplier": friction_multiplier,
        "f_tp": two_phase_friction_factor,
        "frictional_gradient": frictional_gradient,
        "elevational_gradient": elevational_gradient,
        "Ek": acceleration_term,
        "accel_gradient": accel_gradient,
        "total_gradient": total_gradient,
        "V_m": mixture_velocity,
        "V_sl": superficial_liquid_velocity,
        "V_sg": superficial_gas_velocity,
    }

    return total_gradient, details
