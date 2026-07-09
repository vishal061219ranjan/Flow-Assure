from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from backend.utils import safe_ln


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: Productivity Index from Reservoir Properties
# ══════════════════════════════════════════════════════════════════════════════

def calculate_J_from_reservoir(
    k: float,
    h: float,
    mu_o: float,
    Bo: float,
    re: float,
    rw: float,
    s: float = 0.0,
) -> tuple[float, list[str]]:
    """
    Productivity index from radial steady-state Darcy flow.

    Equation
    --------
    J = (k × h) / [141.2 × μo × Bo × (ln(re/rw) − 0.75 + s)]

    Parameters
    ----------
    k    : Permeability, md
    h    : Net pay thickness, ft
    mu_o : Oil viscosity, cP
    Bo   : Oil FVF, RB/STB
    re   : Drainage radius, ft
    rw   : Wellbore radius, ft
    s    : Skin factor (dimensionless)

    Returns
    -------
    tuple[float, list[str]]
        (J in STB/day/psi, warnings)
    """
    warnings: list[str] = []

    if re <= rw:
        warnings.append(f"re ({re}) must be > rw ({rw}); cannot compute J.")
        return 0.0, warnings

    if mu_o <= 0 or Bo <= 0:
        warnings.append("μo and Bo must be positive; cannot compute J.")
        return 0.0, warnings

    ln_term = math.log(re / rw) - 0.75 + s
    if ln_term <= 0:
        warnings.append(
            f"ln(re/rw)−0.75+s = {ln_term:.4f} ≤ 0 (high negative skin?); "
            "clamping denominator to 0.01."
        )
        ln_term = 0.01

    J = (k * h) / (141.2 * mu_o * Bo * ln_term)

    if J <= 0:
        warnings.append("Calculated J ≤ 0; check inputs.")

    return J, warnings


def calculate_J_from_test(
    Pr: float,
    Pwf_test: float,
    qtest: float,
) -> tuple[float, list[str]]:
    """
    Productivity index from a single-point well test (linear IPR assumption).

    Equation
    --------
    J = qtest / (Pr − Pwf_test)

    Returns
    -------
    tuple[float, list[str]]
        (J in STB/day/psi, warnings)
    """
    warnings: list[str] = []
    dp = Pr - Pwf_test
    if dp <= 0:
        warnings.append("Pr − Pwf_test ≤ 0; cannot compute J from test data.")
        return 0.0, warnings

    J = qtest / dp
    return J, warnings


# ══════════════════════════════════════════════════════════════════════════════
#  1. SIMPLE DARCY / PRODUCTIVITY INDEX IPR
# ══════════════════════════════════════════════════════════════════════════════

def darcy_ipr(
    Pr: float,
    J: float,
    N: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate IPR curve using the simple Darcy / Productivity Index model.

    Equation
    --------
    q = J × (Pr − Pwf)

    Parameters
    ----------
    Pr : Average reservoir pressure, psia
    J  : Productivity index, STB/day/psi
    N  : Number of curve points

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame with columns [Pwf, q, Drawdown, Pwf_Pr_ratio, Method]
        dict with key results: qmax, J
    """
    Pwf = np.linspace(Pr, 0, N)
    q = J * (Pr - Pwf)

    df = pd.DataFrame({
        "Pwf (psia)": np.round(Pwf, 2),
        "q (STB/day)": np.round(q, 2),
        "Drawdown (psi)": np.round(Pr - Pwf, 2),
        "Pwf/Pr": np.round(Pwf / Pr, 4) if Pr > 0 else np.zeros(N),
        "Method": "Darcy (PI)",
    })

    qmax = J * Pr

    key_results = {
        "qmax_AOF": round(qmax, 2),
        "J": round(J, 4),
        "method": "Simple Darcy / Productivity Index",
    }

    return df, key_results


# ══════════════════════════════════════════════════════════════════════════════
#  2. VOGEL IPR
# ══════════════════════════════════════════════════════════════════════════════

def calculate_qmax_vogel(
    Pr: float,
    qtest: float,
    Pwf_test: float,
) -> tuple[float, list[str]]:
    """
    Calculate qmax from a test point using the Vogel equation.

    Equation
    --------
    qmax = qtest / [1 − 0.2×(Pwf_test/Pr) − 0.8×(Pwf_test/Pr)²]
    """
    warnings: list[str] = []
    if Pr <= 0:
        warnings.append("Pr must be > 0.")
        return 0.0, warnings

    ratio = Pwf_test / Pr
    denom = 1.0 - 0.2 * ratio - 0.8 * ratio ** 2

    if abs(denom) < 1e-10:
        warnings.append("Vogel denominator ≈ 0 (Pwf_test ≈ Pr); qmax is indeterminate.")
        return 0.0, warnings

    qmax = qtest / denom

    if qmax < 0:
        warnings.append("Calculated qmax < 0; check inputs.")
        qmax = abs(qmax)

    return qmax, warnings


def vogel_ipr(
    Pr: float,
    qmax: float,
    N: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate IPR curve using the **Vogel** model.

    Equation
    --------
    q = qmax × [1 − 0.2×(Pwf/Pr) − 0.8×(Pwf/Pr)²]

    Parameters
    ----------
    Pr   : Reservoir pressure, psia
    qmax : Maximum oil rate (AOF), STB/day
    N    : Number of curve points

    Returns
    -------
    tuple[pd.DataFrame, dict]
    """
    Pwf = np.linspace(Pr, 0, N)
    ratio = Pwf / Pr
    q = qmax * (1.0 - 0.2 * ratio - 0.8 * ratio ** 2)
    q = np.maximum(q, 0)

    df = pd.DataFrame({
        "Pwf (psia)": np.round(Pwf, 2),
        "q (STB/day)": np.round(q, 2),
        "Drawdown (psi)": np.round(Pr - Pwf, 2),
        "Pwf/Pr": np.round(ratio, 4),
        "Method": "Vogel",
    })

    key_results = {
        "qmax_AOF": round(qmax, 2),
        "method": "Vogel IPR",
    }

    return df, key_results


# ══════════════════════════════════════════════════════════════════════════════
#  3. COMPOSITE IPR  (Darcy above Pb + Vogel below Pb)
# ══════════════════════════════════════════════════════════════════════════════


def calculate_J_composite_from_test(
    Pr: float,
    Pb: float,
    qtest: float,
    Pwf_test: float,
) -> tuple[float, list[str]]:
    """
    Calculate J for Composite IPR from a single well-test point.

    CASE 1 — Pwf_test ≥ Pb  (test in undersaturated zone):
        J = q_test / (Pr − Pwf_test)

    CASE 2 — Pwf_test < Pb  (test in saturated zone):
        x_test = Pwf_test / Pb
        J = q_test / [ (Pr − Pb) + (Pb / 1.8) × (1 − 0.2·x_test − 0.8·x_test²) ]

    Validation
    ----------
    • Pr must be > Pb  (otherwise Composite is not applicable).
    • Pb must be > 0.
    • q_test must be > 0.
    • Pwf_test must be between 0 and Pr (exclusive of Pr in Case 1).

    Parameters
    ----------
    Pr       : Reservoir pressure, psia
    Pb       : Bubble-point pressure, psia
    qtest    : Test flow rate, STB/day
    Pwf_test : Flowing BHP at the test point, psia

    Returns
    -------
    tuple[float, list[str]]
        (Productivity index J in STB/day/psi, list of warning/error strings)
    """
    warnings: list[str] = []

    # ── Input validation ──────────────────────────────────────────────────
    if Pb <= 0:
        warnings.append("Composite IPR: Pb must be > 0.")
        return 0.0, warnings

    if Pr <= Pb:
        warnings.append(
            "Composite IPR: Pr must be > Pb. "
            "This method requires an undersaturated reservoir."
        )
        return 0.0, warnings

    if qtest <= 0:
        warnings.append("Composite IPR: q_test must be > 0.")
        return 0.0, warnings

    if Pwf_test < 0 or Pwf_test >= Pr:
        warnings.append(
            f"Composite IPR: Pwf_test ({Pwf_test:.1f}) must be between 0 and Pr ({Pr:.1f})."
        )
        return 0.0, warnings

    # ── CASE 1: Pwf_test ≥ Pb (test in the undersaturated / linear zone) ─
    if Pwf_test >= Pb:
        dP = Pr - Pwf_test                     # drawdown, psi
        if dP <= 0:
            warnings.append("Composite IPR: (Pr − Pwf_test) ≤ 0 — no drawdown.")
            return 0.0, warnings
        J = qtest / dP

    # ── CASE 2: Pwf_test < Pb (test in the saturated / Vogel zone) ───────
    else:
        x_test = Pwf_test / Pb                  # dimensionless pressure ratio
        vogel_term = (Pb / 1.8) * (1.0 - 0.2 * x_test - 0.8 * x_test ** 2)
        denom = (Pr - Pb) + vogel_term
        if denom <= 0:
            warnings.append(
                "Composite IPR: denominator for J back-calculation ≤ 0. "
                "Check Pr, Pb, and Pwf_test values."
            )
            return 0.0, warnings
        J = qtest / denom

    return J, warnings


def composite_ipr(
    Pr: float,
    Pb: float,
    J: float,
    N: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate the full Composite IPR curve once J is known.

    Equations
    ---------
    Bubble-point rate:
        qb = J × (Pr − Pb)

    For Pwf ≥ Pb  (undersaturated / Darcy):
        q = J × (Pr − Pwf)

    For Pwf < Pb  (saturated / Vogel):
        x = Pwf / Pb
        q = qb + (J × Pb / 1.8) × [1 − 0.2·x − 0.8·x²]

    Maximum composite flow rate  (Pwf = 0):
        q_max_composite = qb + (J × Pb / 1.8)

    Physical behaviour
    ------------------
    • The curve is continuous at Pwf = Pb where q = qb.
    • Below Pb the curve bends downward relative to the Darcy extrapolation.
    • At Pwf = 0, q equals q_max_composite.

    Parameters
    ----------
    Pr : Reservoir pressure, psia  (must be > Pb)
    Pb : Bubble-point pressure, psia  (> 0)
    J  : Productivity index, STB/day/psi  (> 0)
    N  : Number of curve points

    Returns
    -------
    tuple[pd.DataFrame, dict]
        DataFrame with columns: Pwf (psia), q (STB/day), Drawdown (psi),
        Pwf/Pr, Method.
        dict with key results: qmax_AOF, qb, Pb, J, method.
    """
    # ── Bubble-point rate ─────────────────────────────────────────────────
    qb = J * (Pr - Pb)

    # ── AOF / maximum composite flow rate (Pwf = 0) ──────────────────────
    qmax = qb + (J * Pb / 1.8)

    # ── Generate pressure points — ensure Pb is included for continuity ──
    #    Split: ~1/3 of points above Pb (linear), ~2/3 below Pb (Vogel)
    n_above = max(N // 3, 5)
    n_below = max(2 * N // 3, 10)
    Pwf_above = np.linspace(Pr, Pb, n_above, endpoint=False)
    Pwf_below = np.linspace(Pb, 0, n_below)
    Pwf = np.concatenate([Pwf_above, Pwf_below])

    # ── Calculate flow rate at each pressure point ────────────────────────
    q = np.zeros_like(Pwf)
    for i, p in enumerate(Pwf):
        if p >= Pb:
            # Undersaturated zone — straight-line Darcy
            q[i] = J * (Pr - p)
        else:
            # Saturated zone — Vogel (with /1.8)
            x = p / Pb
            q[i] = qb + (J * Pb / 1.8) * (1.0 - 0.2 * x - 0.8 * x ** 2)

    q = np.maximum(q, 0)

    # ── Build output DataFrame ────────────────────────────────────────────
    df = pd.DataFrame({
        "Pwf (psia)": np.round(Pwf, 2),
        "q (STB/day)": np.round(q, 2),
        "Drawdown (psi)": np.round(Pr - Pwf, 2),
        "Pwf/Pr": np.round(Pwf / Pr, 4) if Pr > 0 else np.zeros(len(Pwf)),
        "Method": "Composite",
    })

    key_results = {
        "qmax_AOF": round(qmax, 2),
        "qb": round(qb, 2),
        "Pb": Pb,
        "J": round(J, 1),
        "method": "Composite IPR (Darcy + Vogel)",
    }

    return df, key_results


# ══════════════════════════════════════════════════════════════════════════════
#  4. STANDING'S IPR WITH FLOW EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════


def calculate_flow_efficiency(
    r_e: float,
    r_w: float,
    skin: float,
) -> float:
    """Calculate flow efficiency from drainage geometry and skin factor.

    Flow efficiency (FE) quantifies how much a well's actual productivity
    deviates from its ideal (undamaged) productivity.

    Formulas
    --------
    ln_term = ln(0.472 * r_e / r_w)
    FE = ln_term / (ln_term + S)

    Parameters
    ----------
    r_e : float
        Drainage radius in feet.
    r_w : float
        Wellbore radius in feet.
    skin : float
        Skin factor (dimensionless).

    Returns
    -------
    float
        Flow efficiency (dimensionless).

    Raises
    ------
    ValueError
        If r_e or r_w are non-positive, or if the denominator is zero.
    """
    if r_e <= 0.0 or r_w <= 0.0:
        raise ValueError(
            f"Radii must be positive.  Got r_e={r_e}, r_w={r_w}."
        )

    ln_term: float = np.log(0.472 * r_e / r_w)

    denominator: float = ln_term + skin
    if np.isclose(denominator, 0.0):
        raise ValueError(
            "Denominator (ln_term + S) is zero — flow efficiency is "
            "undefined.  Check your skin and radius values."
        )

    fe: float = float(ln_term / denominator)
    return fe


def _modified_pwf(
    p_reference: float,
    pwf: float | np.ndarray,
    fe: float,
) -> float | np.ndarray:
    """Compute the modified (effective) flowing pressure Pwf'.

    Formulas
    --------
    Pwf' = P_ref - FE * (P_ref - Pwf)

    Parameters
    ----------
    p_reference : float
        Reference pressure (Pr for saturated, Pb for undersaturated below
        the bubble point) in psi.
    pwf : float or np.ndarray
        Actual flowing bottom-hole pressure(s) in psi.
    fe : float
        Flow efficiency (dimensionless).

    Returns
    -------
    float or np.ndarray
        Modified flowing pressure Pwf' in psi.
    """
    return p_reference - fe * (p_reference - pwf)


def _vogel_fraction(pwf_prime: float | np.ndarray, p_ref: float) -> float | np.ndarray:
    """Evaluate the Vogel dimensionless IPR bracket.

    Formulas
    --------
    f = 1 - 0.2 * (Pwf' / P_ref) - 0.8 * (Pwf' / P_ref)^2

    Parameters
    ----------
    pwf_prime : float or np.ndarray
        Modified flowing pressure(s) in psi.
    p_ref : float
        Reference pressure for the Vogel equation (Pr or Pb) in psi.

    Returns
    -------
    float or np.ndarray
        Vogel dimensionless fraction (0 to 1).
    """
    ratio = np.asarray(pwf_prime) / p_ref
    return 1.0 - 0.2 * ratio - 0.8 * ratio ** 2


def _solve_saturated(
    pr: float,
    q_test: float,
    pwf_test: float,
    fe_test: float,
    fe_new: float,
    n_points: int,
) -> tuple[pd.DataFrame, dict]:
    """Generate the Standing IPR curve for a **saturated** reservoir (Pr <= Pb).

    Formulas
    --------
    Pwf'_test  = Pr - FE_test * (Pr - Pwf_test)
    qmax_ideal = q_test / [1 - 0.2*(Pwf'_test/Pr) - 0.8*(Pwf'_test/Pr)^2]

    For each Pwf in [Pr … Pwf_min]:
        Pwf'_new = Pr - FE_new * (Pr - Pwf)
        q        = qmax_ideal * [1 - 0.2*(Pwf'_new/Pr) - 0.8*(Pwf'_new/Pr)^2]

    FE > 1 limit: Pwf_min = Pr * (1 - 1/FE) and
        qmax_new = qmax_ideal * (0.624 + 0.376 * FE)
    """
    # Step 1: Effective test pressure
    pwf_prime_test: float = _modified_pwf(pr, pwf_test, fe_test)

    # Step 2: Ideal qmax
    vogel_test: float = float(_vogel_fraction(pwf_prime_test, pr))
    if np.isclose(vogel_test, 0.0):
        raise ValueError(
            "Vogel bracket is zero at the test point — cannot back-calculate "
            "qmax.  The effective test Pwf equals the reservoir pressure."
        )
    qmax_ideal: float = q_test / vogel_test

    # Step 3: Determine valid Pwf range based on FE_new
    if fe_new <= 1.0:
        pwf_min: float = 0.0
        fe_limit_applied: bool = False
        pwf_prime_at_min: float = float(_modified_pwf(pr, 0.0, fe_new))
        qmax_new: float = float(
            qmax_ideal * _vogel_fraction(pwf_prime_at_min, pr)
        )
    else:
        pwf_min = pr * (1.0 - 1.0 / fe_new)
        fe_limit_applied = True
        qmax_new = qmax_ideal * (0.624 + 0.376 * fe_new)

    # Step 4: Generate the IPR curve
    pwf_array: np.ndarray = np.linspace(pr, pwf_min, n_points)
    pwf_prime_array: np.ndarray = _modified_pwf(pr, pwf_array, fe_new)
    pwf_prime_array = np.maximum(pwf_prime_array, 0.0)

    q_array: np.ndarray = qmax_ideal * _vogel_fraction(pwf_prime_array, pr)
    q_array = np.maximum(q_array, 0.0)

    df = pd.DataFrame({
        "Pwf_psi": pwf_array,
        "Pwf_prime_psi": pwf_prime_array,
        "q_o_STBd": q_array,
    })

    results: dict = {
        "reservoir_type": "Saturated (Pr <= Pb)",
        "Pr_psi": pr,
        "Pwf_test_psi": pwf_test,
        "Pwf_prime_test_psi": pwf_prime_test,
        "q_test_STBd": q_test,
        "FE_test": fe_test,
        "FE_new": fe_new,
        "qmax_ideal_STBd": qmax_ideal,
        "qmax_new_STBd": qmax_new,
        "FE_limit_applied": fe_limit_applied,
        "Pwf_min_valid_psi": pwf_min,
    }

    return df, results


def _solve_undersaturated(
    pr: float,
    pb: float,
    q_test: float,
    pwf_test: float,
    fe_test: float,
    fe_new: float,
    n_points: int,
) -> tuple[pd.DataFrame, dict]:
    """Generate the Standing IPR curve for an **undersaturated** reservoir (Pr > Pb).

    Formulas
    --------
    Sub-case B1 (Pwf_test >= Pb — single-phase test):
        J_test = q_test / (Pr - Pwf_test)
        J_new  = J_test * (FE_new / FE_test)

    Sub-case B2 (Pwf_test < Pb — two-phase test):
        Pwf'_test = Pb - FE_test * (Pb - Pwf_test)
        J_test = q_test / { (Pr - Pb)
                 + (Pb/1.8)*[1.8(1-pw_test/pb) - 0.8*FE_test*(1-pw_test/pb)^2] }
        J_new = J_test * (FE_new / FE_test)

    IPR Curve:
        For Pwf >= Pb:  q = J_new * (Pr - Pwf)
        For Pwf < Pb:   Pwf' = Pb - FE_new * (Pb - Pwf)
                        q = J_new*(Pr-Pb) + (J_new*Pb/1.8)*Vogel(Pwf'/Pb)

    FE > 1 limit below Pb: Pwf_min_vogel = Pb * (1 - 1/FE)
    """
    # Determine J_test from the test data
    if pwf_test >= pb:
        sub_case = "B1 (Test Pwf >= Pb, single-phase test)"
        denominator_j: float = pr - pwf_test
        if np.isclose(denominator_j, 0.0):
            raise ValueError(
                "Cannot calculate J: test Pwf equals Pr — no drawdown."
            )
        j_test: float = q_test / denominator_j
    else:
        sub_case = "B2 (Test Pwf < Pb, two-phase test)"
        #New formula Calculation
        ratio: float = pwf_test / pb
        custom_term: float = 1.8 * (1.0 - ratio) - 0.8 * fe_test * ((1.0 - ratio) ** 2)
        denominator_j = (pr - pb) + (pb / 1.8) * custom_term
        if np.isclose(denominator_j, 0.0):
            raise ValueError(
                "Cannot calculate J: composite denominator is zero. "
                "Check input pressures and flow efficiency."
            )
        j_test = q_test / denominator_j

    # Adjust J for the new flow efficiency
    if np.isclose(fe_test, 0.0):
        raise ValueError(
            "FE_test is zero — cannot scale productivity index."
        )
    j_ideal: float = j_test / fe_test
    j_new: float = j_ideal * fe_new
    q_at_pb: float = j_new * (pr - pb)
    qmax_vogel_curve: float = j_new * pb / 1.8
    qmax_vogel_ideal: float = j_ideal * pb / 1.8

    if fe_new <= 1.0:
        pwf_min_vogel: float = 0.0
        fe_limit_applied: bool = False
        
        # New Custom Equation at Pwf = 0 (ratio = 0.0)
        custom_vogel_at_min: float = 1.8 * (1.0 - 0.0) - 0.8 * fe_new * ((1.0 - 0.0) ** 2)
        qmax_total: float = q_at_pb + qmax_vogel_curve * custom_vogel_at_min
    else:
        pwf_min_vogel = pb * (1.0 - 1.0 / fe_new)
        fe_limit_applied = True
        qmax_vogel_corrected: float = qmax_vogel_ideal * (
            0.624 + 0.376 * fe_new
        )
        qmax_total = q_at_pb + qmax_vogel_corrected

    # Generate the composite IPR curve
    pwf_array: np.ndarray = np.linspace(pr, pwf_min_vogel, n_points)

    q_array = np.empty_like(pwf_array)
    pwf_prime_array = np.empty_like(pwf_array)

    for i, pwf in enumerate(pwf_array):
        if pwf >= pb:
            q_array[i] = j_new * (pr - pwf)
            pwf_prime_array[i] = pwf  # No modification above Pb
        else:
            pwf_prime: float = _modified_pwf(pb, pwf, fe_new)
            pwf_prime = max(pwf_prime, 0.0)
            pwf_prime_array[i] = pwf_prime  # Keeping this so your dataframe output doesn't break
            
            # New Custom Equation for each Pwf point
            ratio: float = pwf / pb
            custom_vogel_frac: float = 1.8 * (1.0 - ratio) - 0.8 * fe_new * ((1.0 - ratio) ** 2)
            
            q_array[i] = q_at_pb + qmax_vogel_curve * custom_vogel_frac
    q_array = np.maximum(q_array, 0.0)
    q_bubble: float = j_new * (pr - pb)

    df = pd.DataFrame({
        "Pwf_psi": pwf_array,
        "Pwf_prime_psi": pwf_prime_array,
        "q_o_STBd": q_array,
    })

    results: dict = {
        "reservoir_type": "Undersaturated (Pr > Pb)",
        "sub_case": sub_case,
        "Pr_psi": pr,
        "Pb_psi": pb,
        "Pwf_test_psi": pwf_test,
        "q_test_STBd": q_test,
        "FE_test": fe_test,
        "FE_new": fe_new,
        "J_test_STBd_psi": j_test,
        "J_new_STBd_psi": j_new,
        "q_at_Pb_STBd": q_bubble,
        "qmax_STBd": qmax_total,
        "FE_limit_applied": fe_limit_applied,
        "Pwf_min_valid_psi": pwf_min_vogel,
    }

    return df, results


def standing_ipr(
    pr: float,
    pb: float,
    q_test: float,
    pwf_test: float,
    *,
    fe_old: Optional[float] = None,
    fe_new: float = 1.0,
    r_e: Optional[float] = None,
    r_w: Optional[float] = None,
    skin: Optional[float] = None,
    n_points: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Calculate and tabulate Standing's IPR curve.
    Parameters
    ----------
    pr : float
        Average reservoir pressure (psi).
    pb : float
        Bubble-point pressure (psi).
    q_test : float
        Stabilised test flow rate (STB/d).
    pwf_test : float
        Flowing bottom-hole pressure during the test (psi).
    fe_old : float, optional
        Flow efficiency at the time of the test.  If ``None``, it is
        calculated from ``r_e``, ``r_w``, and ``skin``.
    fe_new : float, default 1.0
        Desired (new) flow efficiency for the generated IPR curve.
    r_e : float, optional
        Drainage radius (ft).  Required if ``fe_old`` is not given.
    r_w : float, optional
        Wellbore radius (ft).  Required if ``fe_old`` is not given.
    skin : float, optional
        Skin factor (dimensionless).  Required if ``fe_old`` is not given.
    n_points : int, default 50
        Number of equally-spaced pressure nodes on the IPR curve.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - **DataFrame** — columns ``Pwf_psi``, ``Pwf_prime_psi``, ``q_o_STBd``.
        - **dict** — key engineering results.

    Raises
    ------
    ValueError
        On missing inputs, non-physical values, or division-by-zero.
    """
    # ---- Input validation ------------------------------------------------
    if pr <= 0:
        raise ValueError(f"Reservoir pressure must be positive.  Got Pr={pr}.")
    if pb < 0:
        raise ValueError(f"Bubble-point pressure cannot be negative.  Got Pb={pb}.")
    if q_test <= 0:
        raise ValueError(f"Test rate must be positive.  Got q_test={q_test}.")
    if pwf_test < 0:
        raise ValueError(f"Flowing pressure cannot be negative.  Got Pwf_test={pwf_test}.")
    if pwf_test >= pr:
        raise ValueError(
            f"Flowing pressure must be less than reservoir pressure.  "
            f"Got Pwf_test={pwf_test} >= Pr={pr}."
        )
    if n_points < 3:
        raise ValueError(f"n_points must be >= 3.  Got {n_points}.")

    # ---- Resolve flow efficiency -----------------------------------------
    if fe_old is not None:
        fe_test: float = fe_old
    elif r_e is not None and r_w is not None and skin is not None:
        fe_test = calculate_flow_efficiency(r_e, r_w, skin)
    else:
        raise ValueError(
            "You must supply either `fe_old` OR all of "
            "`r_e`, `r_w`, and `skin` to determine flow efficiency."
        )

    if fe_test <= 0:
        raise ValueError(
            f"Flow efficiency must be positive.  Got FE_test={fe_test:.4f}."
        )

    # ---- Dispatch to the correct branch ----------------------------------
    if pr <= pb:
        # Branch A: Saturated reservoir
        df, results = _solve_saturated(
            pr=pr,
            q_test=q_test,
            pwf_test=pwf_test,
            fe_test=fe_test,
            fe_new=fe_new,
            n_points=n_points,
        )
    else:
        # Branch B: Undersaturated reservoir
        df, results = _solve_undersaturated(
            pr=pr,
            pb=pb,
            q_test=q_test,
            pwf_test=pwf_test,
            fe_test=fe_test,
            fe_new=fe_new,
            n_points=n_points,
        )

    return df, results


# ══════════════════════════════════════════════════════════════════════════════
#  5. FETKOVICH IPR
# ══════════════════════════════════════════════════════════════════════════════

def generate_fetkovich_ipr(
    Pr: float,
    q_test1: float,
    pwf_test1: float,
    q_test2: Optional[float] = None,
    pwf_test2: Optional[float] = None,
    user_n: Optional[float] = None,
    N: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetkovich IPR calculation based strictly on qo = C * (Pr^2 - Pwf^2)^n.
    """
    # --- Calculating N ---
    # Priority A (User Input)
    if user_n is not None:
        n = float(user_n)
    # Priority B (Two Test Points)
    elif q_test2 is not None and pwf_test2 is not None and q_test2 > 0 and pwf_test2 >= 0:
        delta1 = Pr**2 - pwf_test1**2
        delta2 = Pr**2 - pwf_test2**2
        if delta1 > 0 and delta2 > 0 and q_test1 > 0:
            try:
                n = math.log(q_test1 / q_test2) / math.log(delta1 / delta2)
            except ValueError:
                n = 1.0
        else:
            n = 1.0
    # Priority C (Default)
    else:
        n = 1.0

    # --- Calculating 'C' ---
    delta1 = Pr**2 - pwf_test1**2
    if delta1 > 0:
        C = q_test1 / (delta1 ** n)
    else:
        C = 0.0

    # --- Calculating Absolute Open Flow (qomax) ---
    qomax = C * (Pr**2) ** n

    # --- Calculating Productivity Index (J) [For Knowledge Display Only] ---
    if (Pr - pwf_test1) > 0:
        J = (C * (Pr**2 - pwf_test1**2)**n) / (Pr - pwf_test1)
    else:
        J = 0.0

    # --- Generating the Curve ---
    Pwf_array = np.linspace(Pr, 0.0, N)
    
    q_array = []
    for pwf in Pwf_array:
        delta = Pr**2 - pwf**2
        if delta > 0:
            qo = C * (delta ** n)
        else:
            qo = 0.0
        q_array.append(qo)

    df = pd.DataFrame({
        "Pwf (psia)": np.round(Pwf_array, 2),
        "q (STB/day)": np.round(q_array, 2),
        "Drawdown (psi)": np.round(Pr - Pwf_array, 2),
        "Pwf/Pr": np.round(Pwf_array / Pr, 4) if Pr > 0 else np.zeros(N),
        "Method": f"Fetkovich (n={n:.2f})",
    })

    key_results = {
        "C": C,
        "n": n,
        "qomax": qomax,
        "qmax_AOF": qomax,  # For UI compatibility
        "J": J,
        "method": f"Fetkovich IPR (n = {n:.2f})"
    }

    return df, key_results

#  MASTER IPR GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_ipr_curve(
    method: str,
    params: dict,
) -> tuple[Optional[pd.DataFrame], dict, list[str]]:
    """
    Master function to generate an IPR curve based on the selected method.

    Parameters
    ----------
    method : str
        One of: "Darcy", "Vogel", "Composite", "Standing", "Fetkovich"
    params : dict
        All parameters needed for the selected method.

    Returns
    -------
    tuple[Optional[pd.DataFrame], dict, list[str]]
        (IPR DataFrame, key results dict, warnings)
    """
    warnings: list[str] = []
    N = params.get("N", 50)
    Pr = params.get("Pr")

    if Pr is None or Pr <= 0:
        return None, {}, ["Reservoir pressure (Pr) is required and must be > 0."]

    # ── DARCY ─────────────────────────────────────────────────────────────
    if method == "Darcy":
        darcy_j_source = params.get("darcy_j_source", "test")

        # CASE 1: Manual J
        if darcy_j_source == "manual":
            J = params.get("J")

            if J is None or J <= 0:
                return None, {}, [
                    "Darcy IPR: Manual J selected, but J is missing or invalid."
                ]

            warnings.append(
                f"J entered manually: {J:.4f} STB/day/psi."
            )

        # CASE 2: J from test point
        elif darcy_j_source == "test":
            qtest = params.get("qtest")
            Pwf_test = params.get("Pwf_test")

            if qtest is None or Pwf_test is None or qtest <= 0:
                return None, {}, [
                    "Darcy IPR: Test-point J selected, but qtest or Pwf_test is missing/invalid."
                ]

            J, w = calculate_J_from_test(Pr, Pwf_test, qtest)
            warnings.extend(w)

            if J <= 0:
                return None, {}, [
                    "Darcy IPR: Could not calculate valid J from test point."
                ]

            warnings.append(
                f"J calculated from test point: {J:.4f} STB/day/psi."
            )

        # CASE 3: J from reservoir properties
        elif darcy_j_source == "reservoir":
            k = params.get("k")
            h = params.get("h")
            mu_o = params.get("mu_o")
            Bo = params.get("Bo")
            re = params.get("re", 1000.0)
            rw = params.get("rw", 0.328)
            s = params.get("s", 0.0)

            if all(v is not None and v > 0 for v in [k, h, mu_o, Bo, re, rw]):
                J, w = calculate_J_from_reservoir(
                    k=k,
                    h=h,
                    mu_o=mu_o,
                    Bo=Bo,
                    re=re,
                    rw=rw,
                    s=s,
                )
                warnings.extend(w)

                if J <= 0:
                    return None, {}, [
                        "Darcy IPR: Reservoir-property calculation gave invalid J."
                    ]

                warnings.append(
                    f"J calculated from reservoir properties: {J:.4f} STB/day/psi."
                )

            else:
                return None, {}, [
                    "Darcy IPR: Reservoir-property J selected, but one or more inputs are missing/invalid: "
                    "k, h, μo, Bo, re, rw."
                ]

        # CASE 4: Wrong source name
        else:
            return None, {}, [
                f"Darcy IPR: Unknown darcy_j_source = {darcy_j_source}. "
                "Use 'manual', 'test', or 'reservoir'."
            ]

        df, key_results = darcy_ipr(Pr, J, N)

        key_results["J"] = round(J, 4)
        key_results["J_source"] = darcy_j_source

        return df, key_results, warnings

    # ── VOGEL ─────────────────────────────────────────────────────────────
    elif method == "Vogel":
        Pb = params.get("Pb")
        if Pb is not None and Pr > Pb:
            warnings.append(
                "⚠ Vogel IPR is designed for saturated reservoirs (Pr ≤ Pb). "
                "Your Pr > Pb — consider using Composite IPR instead."
            )

        qmax = params.get("qmax")
        if qmax is None or qmax <= 0:
            qtest = params.get("qtest")
            Pwf_test = params.get("Pwf_test")
            if qtest is not None and Pwf_test is not None and qtest > 0:
                qmax, w = calculate_qmax_vogel(Pr, qtest, Pwf_test)
                warnings.extend(w)
            else:
                return None, {}, [
                    "Cannot compute qmax: provide qmax directly or test point (qtest + Pwf_test)."
                ]

        df, key_results = vogel_ipr(Pr, qmax, N)
        return df, key_results, warnings

    # ── COMPOSITE ─────────────────────────────────────────────────────────
    elif method == "Composite":
        # ── Strict validation per user specification ──────────────────────
        Pb = params.get("Pb")
        if Pb is None or Pb <= 0:
            return None, {}, [
                "Composite IPR: Bubble point pressure (Pb) must be > 0."
            ]

        if Pr <= Pb:
            return None, {}, [
                "Composite IPR: Reservoir pressure (Pr) must be > Pb. "
                "This method requires an undersaturated reservoir."
            ]

        qtest = params.get("qtest")
        Pwf_test = params.get("Pwf_test")

        if qtest is None or qtest <= 0:
            return None, {}, [
                "Composite IPR: Test rate (q_test) must be > 0."
            ]

        if Pwf_test is None or Pwf_test < 0 or Pwf_test >= Pr:
            return None, {}, [
                f"Composite IPR: Pwf_test must be between 0 and Pr ({Pr:.1f} psia)."
            ]

        # ── Calculate J from test point (handles both Case 1 & 2) ────────
        J, w = calculate_J_composite_from_test(Pr, Pb, qtest, Pwf_test)
        warnings.extend(w)

        if J <= 0:
            return None, {}, [
                "Composite IPR: Could not compute a valid J from the given inputs. "
                "Check Pr, Pb, q_test, and Pwf_test."
            ]

        # ── Generate the Composite IPR curve ─────────────────────────────
        df, key_results = composite_ipr(Pr, Pb, J, N)
        return df, key_results, warnings

    # ── STANDING (FE) ─────────────────────────────────────────────────────
    elif method == "Standing":
        qtest = params.get("qtest")
        Pwf_test = params.get("Pwf_test")
        if qtest is None or Pwf_test is None or qtest <= 0:
            return None, {}, [
                "Standing IPR requires a test point (qtest + Pwf_test)."
            ]

        Pb = params.get("Pb")
        if Pb is None or Pb <= 0:
            Pb = Pr  # default: treat as saturated (Pr = Pb)
            warnings.append(
                "Pb not provided; defaulting to Pb = Pr (saturated reservoir)."
            )

        fe_new = params.get("fe_new", 1.0)
        fe_old = params.get("fe_old")
        r_e_param = params.get("re")
        r_w_param = params.get("rw")
        skin_param = params.get("s")

        # Determine FE source
        standing_kwargs: dict = {
            "pr": Pr,
            "pb": Pb,
            "q_test": qtest,
            "pwf_test": Pwf_test,
            "fe_new": fe_new,
            "n_points": N,
        }
        if fe_old is not None and fe_old > 0:
            standing_kwargs["fe_old"] = fe_old
        elif (r_e_param is not None and r_w_param is not None
              and skin_param is not None):
            standing_kwargs["r_e"] = r_e_param
            standing_kwargs["r_w"] = r_w_param
            standing_kwargs["skin"] = skin_param
        else:
            # Default: assume ideal test well (FE_old = 1.0)
            standing_kwargs["fe_old"] = 1.0
            warnings.append(
                "FE_old not provided and skin/radius not available; "
                "defaulting FE_test = 1.0 (ideal test well)."
            )

        try:
            df_standing, standing_results = standing_ipr(**standing_kwargs)
        except ValueError as e:
            return None, {}, [f"Standing IPR error: {e}"]

        # FE warnings
        fe_test_val = standing_results.get("FE_test", 1.0)
        if fe_test_val < 0.5:
            warnings.append("⚠ FE_test < 0.5: very low flow efficiency — check skin factor.")
        if fe_new > 2.0:
            warnings.append("⚠ FE_new > 2.0: unusually high flow efficiency.")
        if standing_results.get("FE_limit_applied", False):
            warnings.append(
                f"⚠ FE > 1 pressure limit applied. Curve terminates at "
                f"Pwf_min = {standing_results.get('Pwf_min_valid_psi', 0):.1f} psia "
                f"(Pwf' cannot go negative)."
            )

        # Normalize output to standard column format for plotting
        Pwf_arr = df_standing["Pwf_psi"].values
        q_arr = df_standing["q_o_STBd"].values
        df_normalized = pd.DataFrame({
            "Pwf (psia)": np.round(Pwf_arr, 2),
            "q (STB/day)": np.round(q_arr, 2),
            "Drawdown (psi)": np.round(Pr - Pwf_arr, 2),
            "Pwf/Pr": np.round(Pwf_arr / Pr, 4) if Pr > 0 else np.zeros(len(Pwf_arr)),
            "Method": f"Standing (FE_new={fe_new})",
        })

        # Build well condition string
        if fe_new < 1.0:
            condition = "Damaged Well (FE < 1)"
        elif fe_new == 1.0:
            condition = "Ideal/Undamaged Well (FE = 1)"
        else:
            condition = "Stimulated Well (FE > 1)"

        # Determine qmax from the new results dict
        qmax_val = standing_results.get(
            "qmax_new_STBd",
            standing_results.get("qmax_STBd", 0)
        )

        key_results = {
            "qmax_AOF": round(float(qmax_val), 2),
            "FE": fe_new,
            "FE_test": round(fe_test_val, 4),
            "well_condition": condition,
            "method": f"Standing IPR (FE_new = {fe_new})",
            "reservoir_type": standing_results.get("reservoir_type", ""),
            "FE_limit_applied": standing_results.get("FE_limit_applied", False),
            "Pwf_min_valid_psi": standing_results.get("Pwf_min_valid_psi", 0),
            # Pass through detailed results
            "standing_details": standing_results,
        }
        if "J_new_STBd_psi" in standing_results:
            key_results["J"] = round(standing_results["J_new_STBd_psi"], 4)
        if "q_at_Pb_STBd" in standing_results:
            key_results["qb"] = round(standing_results["q_at_Pb_STBd"], 2)

        return df_normalized, key_results, warnings

    # ── FETKOVICH ─────────────────────────────────────────────────────────
    elif method == "Fetkovich":
        qtest = params.get("qtest")
        Pwf_test = params.get("Pwf_test")

        if qtest is None or Pwf_test is None or qtest <= 0:
            return None, {}, [
                "Fetkovich IPR requires a test point (qtest + Pwf_test)."
            ]

        # Optional second test point
        q_test2 = params.get("q_test2")
        pwf_test2 = params.get("pwf_test2")

        # Optional user-specified n
        user_n = params.get("user_n")

        try:
            df_fetk, fetk_results = generate_fetkovich_ipr(
                Pr=Pr,
                q_test1=qtest,
                pwf_test1=Pwf_test,
                q_test2=q_test2,
                pwf_test2=pwf_test2,
                user_n=user_n,
                N=N,
            )
        except Exception as e:
            return None, {}, [f"Fetkovich IPR error: {e}"]

        key_results = {
            "qmax_AOF": fetk_results["qomax"],
            "C": fetk_results["C"],
            "n": fetk_results["n"],
            "qomax": fetk_results["qomax"],
            "J": fetk_results["J"],
            "method": fetk_results["method"]
        }

        return df_fetk, key_results, warnings
    else:
        return None, {}, [f"Unknown IPR method: {method}"]
