from __future__ import annotations
from typing import Any


def validate_inputs(inputs: dict) -> tuple[list[str], list[str]]:

    errors: list[str] = []
    warnings: list[str] = []

    def _get(key: str, default=None) -> Any:
        return inputs.get(key, default)

    # ── Reservoir Pressure ────────────────────────────────────────────────
    Pr = _get("Pr")
    if Pr is not None and Pr <= 0:
        errors.append("Reservoir pressure (Pr) must be positive.")

    # ── Flowing BHP (test) ────────────────────────────────────────────────
    Pwf_test = _get("Pwf_test")
    if Pwf_test is not None:
        if Pwf_test < 0:
            errors.append("Flowing BHP (Pwf) must be ≥ 0.")
        if Pr is not None and Pwf_test > Pr:
            errors.append(f"Pwf_test ({Pwf_test}) cannot exceed Pr ({Pr}).")

    # ── Bubble Point Pressure ─────────────────────────────────────────────
    Pb = _get("Pb_manual")
    if Pb is not None and Pb > 0:
        if Pr is not None and Pb > Pr:
            warnings.append(
                f"Pb ({Pb} psia) > Pr ({Pr} psia). "
                "This means reservoir is saturated with free gas. Verify this is correct."
            )

    # ── Temperature ───────────────────────────────────────────────────────
    T = _get("T")
    if T is not None and T <= 0:
        errors.append("Temperature must be positive (°F).")

    # ── Permeability ──────────────────────────────────────────────────────
    k = _get("k")
    if k is not None and k <= 0:
        errors.append("Permeability (k) must be positive.")

    # ── Net pay ───────────────────────────────────────────────────────────
    h = _get("h")
    if h is not None and h <= 0:
        errors.append("Net pay thickness (h) must be positive.")

    # ── Drainage / wellbore radius ────────────────────────────────────────
    re = _get("re")
    rw = _get("rw")
    if re is not None and rw is not None and re <= rw:
        errors.append(f"Drainage radius re ({re}) must be > wellbore radius rw ({rw}).")

    # ── API gravity ───────────────────────────────────────────────────────
    api = _get("api")
    if api is not None:
        if api < 10 or api > 70:
            warnings.append(f"API gravity = {api} is outside typical range [10, 70].")

    # ── Gas specific gravity ──────────────────────────────────────────────
    gas_sg = _get("gas_sg")
    if gas_sg is not None:
        if gas_sg < 0.5 or gas_sg > 1.8:
            warnings.append(f"Gas SG = {gas_sg} is outside typical range [0.5, 1.8].")

    # ── Oil rate (test) ───────────────────────────────────────────────────
    qtest = _get("qtest")
    if qtest is not None and qtest < 0:
        errors.append("Oil test rate (qtest) must be ≥ 0.")

    # ── Water cut ─────────────────────────────────────────────────────────
    fw = _get("water_cut")
    if fw is not None:
        if fw < 0 or fw > 1:
            errors.append("Water cut must be between 0 and 1.")

    # ── Manual PVT overrides ──────────────────────────────────────────────
    Bo_manual = _get("Bo_manual")
    if Bo_manual is not None and Bo_manual > 0 and Bo_manual < 1.0:
        warnings.append(f"Bo = {Bo_manual} < 1.0 RB/STB; this is physically unusual.")

    mu_o_manual = _get("mu_o_manual")
    if mu_o_manual is not None and mu_o_manual <= 0:
        errors.append("Oil viscosity must be positive.")

    # ── Flow Efficiency ───────────────────────────────────────────────────
    FE = _get("FE")
    if FE is not None and FE <= 0:
        errors.append("Flow efficiency (FE) must be > 0.")

    # ── Fetkovich n ───────────────────────────────────────────────────────
    n_fetk = _get("n_fetkovich")
    if n_fetk is not None:
        if n_fetk < 0.5 or n_fetk > 1.0:
            warnings.append(
                f"Fetkovich exponent n = {n_fetk} outside recommended range [0.5, 1.0]. "
                "It will be clamped."
            )

    # ── GOR ───────────────────────────────────────────────────────────────
    GOR = _get("GOR")
    if GOR is not None and GOR < 0:
        errors.append("GOR must be ≥ 0.")

    return errors, warnings


def check_ipr_suitability(method: str, inputs: dict) -> list[str]:
    """
    Check if the selected IPR method is suitable for the given inputs.

    Returns a list of suitability warnings (non-blocking).
    """
    suitability: list[str] = []
    Pr = inputs.get("Pr")
    Pb = inputs.get("Pb_manual") or inputs.get("Pb_calculated")


    if method == "Composite":
        if Pb is not None and Pr is not None and Pr <= Pb:
            suitability.append(
                "🔶 Composite IPR is designed for undersaturated reservoirs (Pr > Pb). "
                "Your reservoir is saturated (Pr ≤ Pb). "
                "The curve will behave like pure Vogel."
            )

    elif method == "Darcy":
        if Pb is not None and Pr is not None and Pr <= Pb:
            suitability.append(
                "🔶 Simple Darcy IPR assumes single-phase flow. "
                "Your reservoir is below bubble point — two-phase effects are expected. "
                "Consider using **Vogel** or **Composite IPR**."
            )

    return suitability
