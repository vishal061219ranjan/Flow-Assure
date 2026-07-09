import math
import warnings
from typing import Optional



# ──────────────────────────────────────────────────────────────────────────────
# Physical / Engineering Constants
# ──────────────────────────────────────────────────────────────────────────────

R_UNIVERSAL = 10.732  # Universal gas constant, psia·ft³/(lbmol·°R)
WATER_SG = 1.0        # Default water specific gravity


# ──────────────────────────────────────────────────────────────────────────────
# Unit Conversion Helpers
# ──────────────────────────────────────────────────────────────────────────────

def api_to_sg(api: float) -> float:
    """
    Convert API gravity to oil specific gravity (relative to water).

    Parameters
    ----------
    api : float
        Oil API gravity, °API.

    Returns
    -------
    float
        Oil specific gravity (dimensionless, water = 1.0).

    Formula
    -------
    γo = 141.5 / (API + 131.5)
    """
    if api <= -131.5:
        raise ValueError(f"API gravity must be > -131.5; got {api}")
    return 141.5 / (api + 131.5)


def sg_to_api(sg: float) -> float:
    """
    Convert oil specific gravity to API gravity.

    Parameters
    ----------
    sg : float
        Oil specific gravity (water = 1.0).

    Returns
    -------
    float
        Oil API gravity, °API.

    Formula
    -------
    API = (141.5 / γo) − 131.5
    """
    if sg <= 0:
        raise ValueError(f"Specific gravity must be positive; got {sg}")
    return (141.5 / sg) - 131.5


def fahrenheit_to_rankine(T_F: float) -> float:
    """Convert temperature from °F to °R (Rankine)."""
    return T_F + 459.67


def psi_to_atm(psi: float) -> float:
    """Convert pressure from psia to atm."""
    return psi / 14.696


# ──────────────────────────────────────────────────────────────────────────────
# Gas Pseudo-Critical Properties (Sutton, 1985)
# ──────────────────────────────────────────────────────────────────────────────

def pseudo_critical_properties(gas_sg: float) -> tuple[float, float]:
    """
    Compute pseudo-critical temperature and pressure for a natural gas
    using Sutton (1985) correlations.

    Parameters
    ----------
    gas_sg : float
        Gas specific gravity (air = 1.0).

    Returns
    -------
    tuple[float, float]
        (Tpc in °R, Ppc in psia)

    Equations
    ---------
    Tpc = 169.2 + 349.5×γg − 74.0×γg²
    Ppc = 756.8 − 131.0×γg − 3.6×γg²
    """
    Tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg ** 2
    Ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg ** 2
    return Tpc, Ppc


def gas_molecular_weight(gas_sg: float) -> float:
    """
    Compute apparent molecular weight of gas from specific gravity.

    Mg = 28.97 × γg
    """
    return 28.97 * gas_sg


# ──────────────────────────────────────────────────────────────────────────────
# Result Tagging Helper
# ──────────────────────────────────────────────────────────────────────────────

class PVTResult:
    """Container for a single PVT calculation result with provenance tag."""

    def __init__(
        self,
        name: str,
        symbol: str,
        value: Optional[float],
        unit: str,
        source: str,
        warnings: Optional[list[str]] = None,
    ):
        self.name = name
        self.symbol = symbol
        self.value = value
        self.unit = unit
        self.source = source          # e.g. "User Input", "Standing Correlation"
        self.warnings = warnings or []

    def __repr__(self) -> str:
        w = f" ⚠ {'; '.join(self.warnings)}" if self.warnings else ""
        val = f"{self.value:.4f}" if self.value is not None else "N/A"
        return f"{self.name} ({self.symbol}) = {val} {self.unit} [{self.source}]{w}"

    def to_dict(self) -> dict:
        return {
            "Property": self.name,
            "Symbol": self.symbol,
            "Value": round(self.value, 4) if self.value is not None else None,
            "Unit": self.unit,
            "Source": self.source,
            "Warnings": "; ".join(self.warnings) if self.warnings else "",
        }


def safe_log10(x):
    if x <= 0:
        warnings.warn(f"safe_log10 received an invalid input: {x}. Returning 0.0", UserWarning)
        return 0.0
    return math.log10(x)
def safe_ln(x):
    if x <= 0:
        warnings.warn(f"safe_ln received an invalid input: {x}. Returning 0.0", UserWarning)
        return 0.0
    return math.log(x)
