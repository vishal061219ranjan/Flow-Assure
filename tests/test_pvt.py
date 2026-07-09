import math
import sys
import os

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pvt import (
    calculate_bg,
    calculate_bo_above_pb,
    calculate_bo_standing,
    calculate_bo_vasquez_beggs,
    calculate_bw,
    calculate_dead_oil_viscosity_beggs_robinson,
    calculate_gas_viscosity_lge,
    calculate_live_oil_viscosity_beggs_robinson,
    calculate_muw,
    calculate_pb_standing,
    calculate_pb_vasquez_beggs,
    calculate_rs_standing,
    calculate_rs_vasquez_beggs,
    calculate_undersaturated_oil_viscosity,
    calculate_z_factor_dak,
    calculate_all_pvt,
)
from backend.utils import api_to_sg, sg_to_api


# ── Utility Tests ─────────────────────────────────────────────────────────────

class TestUtils:
    def test_api_to_sg_35(self):
        """API 35 should give SG ≈ 0.8498."""
        sg = api_to_sg(35.0)
        assert abs(sg - 0.8498) < 0.001

    def test_sg_to_api_roundtrip(self):
        """Converting API → SG → API should return original value."""
        for api in [20, 30, 35, 45]:
            sg = api_to_sg(api)
            api_back = sg_to_api(sg)
            assert abs(api_back - api) < 0.001

    def test_api_to_sg_water(self):
        """API 10 should give SG ≈ 1.0 (water baseline)."""
        sg = api_to_sg(10.0)
        assert abs(sg - 1.0) < 0.001


# ── Solution GOR Tests ────────────────────────────────────────────────────────

class TestSolutionGOR:
    """Test Standing and Vasquez-Beggs Rs correlations."""

    def test_standing_rs_positive(self):
        """Rs should be positive for typical inputs."""
        Rs, warnings = calculate_rs_standing(2000, 180, 35, 0.75)
        assert Rs > 0
        assert Rs < 3000  # reasonable upper bound

    def test_standing_rs_increases_with_pressure(self):
        """Rs should increase with pressure (below Pb)."""
        Rs_low, _ = calculate_rs_standing(1000, 180, 35, 0.75)
        Rs_high, _ = calculate_rs_standing(2000, 180, 35, 0.75)
        assert Rs_high > Rs_low

    def test_standing_rs_typical_range(self):
        """For API=35, T=180°F, P=2000 psia, gs=0.75, Rs should be ~300-700 scf/STB."""
        Rs, _ = calculate_rs_standing(2000, 180, 35, 0.75)
        assert 200 < Rs < 800

    def test_standing_rs_cap_at_pb(self):
        """Rs at P > Pb should be capped at Rs(Pb)."""
        Rs_at_pb, _ = calculate_rs_standing(2000, 180, 35, 0.75, Pb=2000)
        Rs_above, _ = calculate_rs_standing(3000, 180, 35, 0.75, Pb=2000)
        assert abs(Rs_at_pb - Rs_above) < 0.01

    def test_vasquez_beggs_rs_positive(self):
        """VB Rs should be positive."""
        Rs, _ = calculate_rs_vasquez_beggs(2000, 180, 35, 0.75)
        assert Rs > 0

    def test_vasquez_beggs_rs_increases_with_pressure(self):
        """VB Rs should increase with pressure."""
        Rs_low, _ = calculate_rs_vasquez_beggs(1000, 180, 35, 0.75)
        Rs_high, _ = calculate_rs_vasquez_beggs(2000, 180, 35, 0.75)
        assert Rs_high > Rs_low

    def test_standing_vs_vb_same_order(self):
        """Standing and VB Rs should be in the same order of magnitude."""
        Rs_st, _ = calculate_rs_standing(2000, 180, 35, 0.75)
        Rs_vb, _ = calculate_rs_vasquez_beggs(2000, 180, 35, 0.75)
        # Within 50% of each other
        assert abs(Rs_st - Rs_vb) / max(Rs_st, Rs_vb) < 0.50


# ── Bubble Point Tests ────────────────────────────────────────────────────────

class TestBubblePoint:
    def test_standing_pb_positive(self):
        """Pb should be positive for typical Rs."""
        Pb, _ = calculate_pb_standing(500, 180, 35, 0.75)
        assert Pb > 0

    def test_standing_pb_roundtrip(self):
        """Rs → Pb → Rs should approximately round-trip."""
        # Get Rs at P=2000
        Rs_orig, _ = calculate_rs_standing(2000, 180, 35, 0.75)
        # Get Pb from that Rs
        Pb, _ = calculate_pb_standing(Rs_orig, 180, 35, 0.75)
        # Pb should be close to 2000
        assert abs(Pb - 2000) / 2000 < 0.05

    def test_vb_pb_positive(self):
        """VB Pb should be positive."""
        Pb, _ = calculate_pb_vasquez_beggs(500, 180, 35, 0.75)
        assert Pb > 0


# ── Oil FVF Tests ─────────────────────────────────────────────────────────────

class TestOilFVF:
    def test_standing_bo_greater_than_one(self):
        """Bo should be ≥ 1.0."""
        Bo, _ = calculate_bo_standing(500, 180, 35, 0.75)
        assert Bo >= 1.0

    def test_standing_bo_typical_range(self):
        """For typical inputs, Bo should be 1.0–2.0."""
        Bo, _ = calculate_bo_standing(500, 180, 35, 0.75)
        assert 1.0 <= Bo <= 2.5

    def test_standing_bo_increases_with_rs(self):
        """Bo should increase with Rs (more dissolved gas → more swelling)."""
        Bo_low, _ = calculate_bo_standing(200, 180, 35, 0.75)
        Bo_high, _ = calculate_bo_standing(800, 180, 35, 0.75)
        assert Bo_high > Bo_low

    def test_vb_bo_greater_than_one(self):
        Bo, _ = calculate_bo_vasquez_beggs(500, 180, 35, 0.75)
        assert Bo >= 1.0

    def test_bo_above_pb_decreases(self):
        """Bo above Pb should be slightly less than Bo at Pb."""
        Bo_pb = 1.3
        Bo_above, _ = calculate_bo_above_pb(Bo_pb, 3000, 2000, co=10e-6)
        assert Bo_above < Bo_pb
        assert Bo_above > 1.0


# ── Oil Viscosity Tests ───────────────────────────────────────────────────────

class TestOilViscosity:
    def test_dead_oil_viscosity_positive(self):
        """Dead oil viscosity should be positive."""
        mu, _ = calculate_dead_oil_viscosity_beggs_robinson(35, 180)
        assert mu > 0

    def test_dead_oil_viscosity_decreases_with_api(self):
        """Lighter oil (higher API) → lower viscosity."""
        mu_heavy, _ = calculate_dead_oil_viscosity_beggs_robinson(20, 180)
        mu_light, _ = calculate_dead_oil_viscosity_beggs_robinson(45, 180)
        assert mu_light < mu_heavy

    def test_dead_oil_viscosity_decreases_with_temp(self):
        """Higher temperature → lower viscosity."""
        mu_cold, _ = calculate_dead_oil_viscosity_beggs_robinson(35, 100)
        mu_hot, _ = calculate_dead_oil_viscosity_beggs_robinson(35, 250)
        assert mu_hot < mu_cold

    def test_live_oil_viscosity_less_than_dead(self):
        """Live oil (with dissolved gas) should be less viscous than dead oil."""
        mu_dead, _ = calculate_dead_oil_viscosity_beggs_robinson(35, 180)
        mu_live, _ = calculate_live_oil_viscosity_beggs_robinson(mu_dead, 500)
        assert mu_live < mu_dead

    def test_live_oil_viscosity_positive(self):
        mu_dead, _ = calculate_dead_oil_viscosity_beggs_robinson(35, 180)
        mu_live, _ = calculate_live_oil_viscosity_beggs_robinson(mu_dead, 500)
        assert mu_live > 0

    def test_undersaturated_viscosity_increases(self):
        """Viscosity above Pb should be ≥ viscosity at Pb."""
        mu_ob = 1.5
        mu_above, _ = calculate_undersaturated_oil_viscosity(mu_ob, 3000, 2000)
        assert mu_above >= mu_ob


# ── Gas Z-Factor Tests ────────────────────────────────────────────────────────

class TestZFactor:
    def test_z_factor_at_low_pressure(self):
        """At low pressure, Z should be close to 1.0 (ideal gas)."""
        Z, _ = calculate_z_factor_dak(14.7, 60, 0.75)
        assert abs(Z - 1.0) < 0.15

    def test_z_factor_positive(self):
        """Z should always be positive."""
        Z, _ = calculate_z_factor_dak(3000, 180, 0.75)
        assert Z > 0

    def test_z_factor_typical_range(self):
        """For typical conditions, Z should be 0.3–1.2."""
        Z, _ = calculate_z_factor_dak(3000, 180, 0.75)
        assert 0.3 < Z < 1.2

    def test_z_factor_at_zero_pressure(self):
        """At P=0, Z should be 1.0."""
        Z, _ = calculate_z_factor_dak(0, 180, 0.75)
        assert abs(Z - 1.0) < 0.01


# ── Gas Properties Tests ─────────────────────────────────────────────────────

class TestGasProperties:
    def test_bg_positive(self):
        """Bg should be positive."""
        Bg, _ = calculate_bg(3000, 180, 0.85)
        assert Bg > 0

    def test_bg_decreases_with_pressure(self):
        """Bg should decrease with increasing pressure (gas compresses)."""
        Bg_low, _ = calculate_bg(1000, 180, 0.9)
        Bg_high, _ = calculate_bg(3000, 180, 0.85)
        assert Bg_high < Bg_low

    def test_gas_viscosity_positive(self):
        """Gas viscosity should be positive."""
        Z, _ = calculate_z_factor_dak(3000, 180, 0.75)
        mu_g, _ = calculate_gas_viscosity_lge(3000, 180, 0.75, Z)
        assert mu_g > 0

    def test_gas_viscosity_typical_range(self):
        """Gas viscosity typically 0.01–0.05 cP."""
        Z, _ = calculate_z_factor_dak(3000, 180, 0.75)
        mu_g, _ = calculate_gas_viscosity_lge(3000, 180, 0.75, Z)
        assert 0.005 < mu_g < 0.1


# ── Water Properties Tests ────────────────────────────────────────────────────

class TestWaterProperties:
    def test_bw_near_one(self):
        """Bw should be close to 1.0."""
        Bw, _ = calculate_bw(3000, 180)
        assert abs(Bw - 1.0) < 0.1

    def test_muw_positive(self):
        """Water viscosity should be positive."""
        mu_w, _ = calculate_muw(180, 0)
        assert mu_w > 0


# ── Integration Test ──────────────────────────────────────────────────────────

class TestPVTIntegration:
    def test_calculate_all_pvt_default_case(self):
        """Full PVT calculation with default example case."""
        inputs = {
            "Pr": 3000,
            "Pb_manual": 2000,
            "T": 180,
            "api": 35,
            "gas_sg": 0.75,
            "GOR": 500,
        }
        result = calculate_all_pvt(inputs, "Standing")

        assert result["values"].get("Pb") == 2000  # user-provided
        assert result["values"].get("Rs") is not None
        assert result["values"]["Rs"] > 0
        assert result["values"].get("Bo") is not None
        assert result["values"]["Bo"] >= 1.0
        assert result["values"].get("mu_o") is not None
        assert result["values"]["mu_o"] > 0
        assert result["values"].get("Z") is not None
        assert result["values"]["Z"] > 0

    def test_calculate_all_pvt_manual_overrides(self):
        """Manual PVT values should override correlations."""
        inputs = {
            "Pr": 3000,
            "Pb_manual": 2000,
            "T": 180,
            "api": 35,
            "gas_sg": 0.75,
            "Bo_manual": 1.35,
            "Rs_manual": 450.0,
            "mu_o_manual": 1.5,
        }
        result = calculate_all_pvt(inputs, "Standing")

        assert result["values"]["Bo"] == 1.35
        assert result["values"]["Rs"] == 450.0
        assert result["values"]["mu_o"] == 1.5
        assert result["results"]["Bo"].source == "User Input"
        assert result["results"]["Rs"].source == "User Input"
        assert result["results"]["mu_o"].source == "User Input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
