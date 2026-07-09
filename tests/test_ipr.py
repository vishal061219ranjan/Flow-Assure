
import math
import sys
import os

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ipr import (
    calculate_flow_efficiency,
    calculate_J_composite_from_test,
    calculate_J_from_reservoir,
    calculate_J_from_test,
    calculate_qmax_vogel,
    composite_ipr,
    darcy_ipr,
    generate_fetkovich_ipr,
    generate_ipr_curve,
    standing_ipr,
    vogel_ipr,
)


# ══════════════════════════════════════════════════════════════════════════════
#  J Calculation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestProductivityIndex:
    def test_J_from_test_data(self):
        """J = q / (Pr - Pwf) should give correct result."""
        J, warnings = calculate_J_from_test(3000, 1500, 800)
        expected = 800 / (3000 - 1500)
        assert abs(J - expected) < 0.001

    def test_J_from_reservoir_positive(self):
        """J from reservoir properties should be positive."""
        J, _ = calculate_J_from_reservoir(
            k=50, h=40, mu_o=1.5, Bo=1.3, re=1000, rw=0.328, s=0)
        assert J > 0

    def test_J_from_reservoir_with_skin(self):
        """Positive skin should decrease J."""
        J_no_skin, _ = calculate_J_from_reservoir(50, 40, 1.5, 1.3, 1000, 0.328, 0)
        J_skin, _ = calculate_J_from_reservoir(50, 40, 1.5, 1.3, 1000, 0.328, 5)
        assert J_skin < J_no_skin

    def test_J_from_reservoir_negative_skin(self):
        """Negative skin should increase J."""
        J_no_skin, _ = calculate_J_from_reservoir(50, 40, 1.5, 1.3, 1000, 0.328, 0)
        J_neg_skin, _ = calculate_J_from_reservoir(50, 40, 1.5, 1.3, 1000, 0.328, -2)
        assert J_neg_skin > J_no_skin


# ══════════════════════════════════════════════════════════════════════════════
#  Darcy IPR Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDarcyIPR:
    def test_q_at_Pr_is_zero(self):
        """At Pwf = Pr, flow rate should be zero."""
        df, results = darcy_ipr(3000, 1.0, 50)
        q_at_Pr = df[df["Pwf (psia)"] == 3000.0]["q (STB/day)"].values[0]
        assert abs(q_at_Pr) < 0.1

    def test_q_at_zero_Pwf(self):
        """At Pwf = 0, q should equal J × Pr."""
        J = 1.5
        Pr = 3000
        df, results = darcy_ipr(Pr, J, 50)
        q_at_zero = df[df["Pwf (psia)"] == 0.0]["q (STB/day)"].values[0]
        assert abs(q_at_zero - J * Pr) < 1.0

    def test_qmax_equals_J_times_Pr(self):
        """Key result qmax should equal J × Pr."""
        J = 1.5
        Pr = 3000
        _, results = darcy_ipr(Pr, J, 50)
        assert abs(results["qmax_AOF"] - J * Pr) < 1.0

    def test_darcy_linearity(self):
        """Darcy IPR should be linear — constant J throughout."""
        J = 2.0
        Pr = 3000
        df, _ = darcy_ipr(Pr, J, 100)
        # Check several points
        for _, row in df.iterrows():
            expected_q = J * row["Drawdown (psi)"]
            assert abs(row["q (STB/day)"] - expected_q) < 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Vogel IPR Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestVogelIPR:
    def test_q_at_Pr_is_zero(self):
        """At Pwf = Pr, Vogel gives q = 0."""
        df, _ = vogel_ipr(3000, 2000, 50)
        q_at_Pr = df[df["Pwf (psia)"] == 3000.0]["q (STB/day)"].values[0]
        assert abs(q_at_Pr) < 0.1

    def test_q_at_zero_equals_qmax(self):
        """At Pwf = 0, Vogel gives q = qmax."""
        qmax = 2000
        df, results = vogel_ipr(3000, qmax, 50)
        q_at_zero = df[df["Pwf (psia)"] == 0.0]["q (STB/day)"].values[0]
        assert abs(q_at_zero - qmax) < 1.0

    def test_qmax_from_test_point(self):
        """qmax calculation from test data."""
        Pr = 3000
        Pwf_test = 1500
        qtest = 800
        qmax, _ = calculate_qmax_vogel(Pr, qtest, Pwf_test)

        # Verify: plugging qmax back should give qtest at Pwf_test
        ratio = Pwf_test / Pr
        q_check = qmax * (1 - 0.2 * ratio - 0.8 * ratio ** 2)
        assert abs(q_check - qtest) < 1.0

    def test_vogel_curve_monotonic(self):
        """Vogel curve should be monotonically decreasing in Pwf (q increases as Pwf decreases)."""
        df, _ = vogel_ipr(3000, 2000, 50)
        q_values = df["q (STB/day)"].values
        # q should increase as we go from top (Pwf=Pr) to bottom (Pwf=0)
        for i in range(len(q_values) - 1):
            assert q_values[i] <= q_values[i + 1] + 0.1  # allowing small rounding tolerance


# ══════════════════════════════════════════════════════════════════════════════
#  Composite IPR Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeIPR:
    def test_continuity_at_bubble_point(self):
        """Composite IPR must be continuous at Pb."""
        Pr = 3000
        Pb = 2000
        J = 1.0
        df, _ = composite_ipr(Pr, Pb, J, 200)

        # Find points closest to Pb from both sides
        above = df[df["Pwf (psia)"] >= Pb].iloc[-1]
        below = df[df["Pwf (psia)"] <= Pb].iloc[0]

        q_above = above["q (STB/day)"]
        q_below = below["q (STB/day)"]

        # They should be very close (within 1 STB/day)
        assert abs(q_above - q_below) < 5.0

    def test_qb_equals_J_times_drawdown(self):
        """qb should equal J × (Pr - Pb)."""
        Pr = 3000
        Pb = 2000
        J = 1.5
        _, results = composite_ipr(Pr, Pb, J, 50)
        expected_qb = J * (Pr - Pb)
        assert abs(results["qb"] - expected_qb) < 1.0

    def test_qmax_aof(self):
        """AOF should equal qb + J×Pb/1.8."""
        Pr = 3000
        Pb = 2000
        J = 1.5
        _, results = composite_ipr(Pr, Pb, J, 50)
        expected_aof = J * (Pr - Pb) + J * Pb / 1.8
        assert abs(results["qmax_AOF"] - expected_aof) < 1.0

    def test_q_at_Pr_is_zero(self):
        """At Pwf = Pr, q should be zero."""
        df, _ = composite_ipr(3000, 2000, 1.0, 50)
        q_at_Pr = df[df["Pwf (psia)"] >= 2999]["q (STB/day)"].values[0]
        assert abs(q_at_Pr) < 1.0

    def test_J_from_test_above_pb(self):
        """J from test point above Pb."""
        J, _ = calculate_J_composite_from_test(3000, 2000, 500, 2500)
        expected = 500 / (3000 - 2500)
        assert abs(J - expected) < 0.001

    def test_J_from_test_below_pb(self):
        """J from test point below Pb should be positive."""
        J, _ = calculate_J_composite_from_test(3000, 2000, 800, 1500)
        assert J > 0


# ══════════════════════════════════════════════════════════════════════════════
#  Standing IPR (New Module) Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFlowEfficiency:
    def test_zero_skin_ideal(self):
        """Zero skin should give FE ≈ 1.0."""
        fe = calculate_flow_efficiency(r_e=1000, r_w=0.328, skin=0.0)
        assert abs(fe - 1.0) < 0.001

    def test_positive_skin_damaged(self):
        """Positive skin should give FE < 1.0."""
        fe = calculate_flow_efficiency(r_e=1000, r_w=0.328, skin=5.0)
        assert fe < 1.0
        assert fe > 0.0

    def test_negative_skin_stimulated(self):
        """Negative skin should give FE > 1.0."""
        fe = calculate_flow_efficiency(r_e=1000, r_w=0.328, skin=-2.0)
        assert fe > 1.0


class TestStandingIPR:
    """Tests for the new standing_ipr function."""

    def test_saturated_basic(self):
        """Saturated reservoir with FE=1 should produce a valid curve."""
        df, res = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=1.0, n_points=25,
        )
        assert len(df) == 25
        assert "Pwf_psi" in df.columns
        assert "q_o_STBd" in df.columns
        assert res["reservoir_type"] == "Saturated (Pr <= Pb)"
        assert res["qmax_new_STBd"] > 0

    def test_undersaturated_basic(self):
        """Undersaturated reservoir should produce a valid curve."""
        df, res = standing_ipr(
            pr=4000, pb=2500, q_test=800, pwf_test=3200,
            fe_old=1.0, fe_new=1.0, n_points=25,
        )
        assert len(df) == 25
        assert "Undersaturated" in res["reservoir_type"]
        assert res["qmax_STBd"] > 0
        assert res["J_new_STBd_psi"] > 0

    def test_damaged_well_produces_less(self):
        """Damaged well (FE_new < 1) should produce less than ideal."""
        df_ideal, _ = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=1.0, n_points=50,
        )
        df_damaged, _ = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=0.7, n_points=50,
        )
        q_ideal_mid = df_ideal.iloc[len(df_ideal) // 2]["q_o_STBd"]
        q_damaged_mid = df_damaged.iloc[len(df_damaged) // 2]["q_o_STBd"]
        assert q_damaged_mid < q_ideal_mid

    def test_stimulated_well_produces_more(self):
        """Stimulated well (FE_new > 1) should produce higher qmax than ideal."""
        _, res_ideal = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=1.0, n_points=50,
        )
        _, res_stim = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=1.4, n_points=50,
        )
        # Stimulated well should have higher maximum rate
        assert res_stim["qmax_new_STBd"] > res_ideal["qmax_new_STBd"]

    def test_fe_gt1_limit_saturated(self):
        """FE > 1 on saturated reservoir should apply pressure limit."""
        _, res = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=1.4, n_points=26,
        )
        assert res["FE_limit_applied"] is True
        assert res["Pwf_min_valid_psi"] > 0
        expected_min = 2500 * (1 - 1 / 1.4)
        assert abs(res["Pwf_min_valid_psi"] - expected_min) < 1.0

    def test_fe_gt1_limit_undersaturated(self):
        """FE > 1 on undersaturated reservoir should apply Vogel section limit."""
        _, res = standing_ipr(
            pr=4000, pb=2500, q_test=800, pwf_test=3200,
            fe_old=1.0, fe_new=1.3, n_points=26,
        )
        assert res["FE_limit_applied"] is True
        expected_min = 2500 * (1 - 1 / 1.3)
        assert abs(res["Pwf_min_valid_psi"] - expected_min) < 1.0

    def test_fe_from_skin_radius(self):
        """FE should be calculable from skin and radius."""
        df, res = standing_ipr(
            pr=4000, pb=2500, q_test=800, pwf_test=3200,
            r_e=1490, r_w=0.328, skin=5.0, fe_new=1.0, n_points=25,
        )
        assert res["FE_test"] < 1.0
        assert df is not None

    def test_q_zero_at_pr(self):
        """At Pwf = Pr, flow rate should be zero."""
        df, _ = standing_ipr(
            pr=2500, pb=2500, q_test=500, pwf_test=1800,
            fe_old=1.0, fe_new=1.0, n_points=25,
        )
        assert abs(df.iloc[0]["q_o_STBd"]) < 0.1

    def test_invalid_inputs_raise(self):
        """Invalid inputs should raise ValueError."""
        with pytest.raises(ValueError):
            standing_ipr(pr=-100, pb=2500, q_test=500, pwf_test=1800, fe_old=1.0)
        with pytest.raises(ValueError):
            standing_ipr(pr=3000, pb=2500, q_test=500, pwf_test=3000, fe_old=1.0)
        with pytest.raises(ValueError):
            standing_ipr(pr=3000, pb=2500, q_test=500, pwf_test=1800)  # no FE source



# ══════════════════════════════════════════════════════════════════════════════
#  Fetkovich IPR Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFetkovichIPR:
    """Tests for the new simplified generate_fetkovich_ipr function."""

    def test_single_point_default(self):
        """Single test point with default n=1.0 should produce a valid curve."""
        df, res = generate_fetkovich_ipr(
            Pr=2500, q_test1=600, pwf_test1=1800,
            N=25,
        )
        assert len(df) == 25
        assert "Pwf (psia)" in df.columns
        assert "q (STB/day)" in df.columns
        assert res["qomax"] > 0
        assert res["n"] == 1.0
        assert res["C"] > 0
        assert res["J"] > 0

    def test_two_test_points_n_calculation(self):
        """Two test points should calculate n analytically."""
        df, res = generate_fetkovich_ipr(
            Pr=3000,
            q_test1=800, pwf_test1=1500,
            q_test2=1200, pwf_test2=1000,
            N=25,
        )
        assert res["n"] != 1.0
        assert res["C"] > 0
        assert res["qomax"] > 0

    def test_user_specified_n(self):
        """User-specified n should be used."""
        df, res = generate_fetkovich_ipr(
            Pr=3000, q_test1=800, pwf_test1=1500,
            user_n=0.85, N=25,
        )
        assert res["n"] == 0.85

    def test_q_at_Pr_is_zero(self):
        """At Pwf = Pr, flow rate should be zero."""
        df, _ = generate_fetkovich_ipr(
            Pr=3000, q_test1=800, pwf_test1=1500,
            N=50,
        )
        q_at_Pr = df.iloc[0]["q (STB/day)"]
        assert abs(q_at_Pr) < 1.0

    def test_invalid_inputs_handled(self):
        """Invalid inputs should be handled gracefully (C=0, q=0)."""
        df, res = generate_fetkovich_ipr(Pr=-100, q_test1=800, pwf_test1=1500)
        assert res["C"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Master generate_ipr_curve Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateIPRCurve:
    def test_darcy_with_test_point(self):
        """Generate Darcy IPR from test point."""
        params = {"Pr": 3000, "qtest": 800, "Pwf_test": 1500, "N": 50}
        df, results, warnings = generate_ipr_curve("Darcy", params)
        assert df is not None
        assert results["qmax_AOF"] > 0

    def test_vogel_with_test_point(self):
        """Generate Vogel IPR from test point."""
        params = {"Pr": 3000, "qtest": 800, "Pwf_test": 1500, "N": 50}
        df, results, warnings = generate_ipr_curve("Vogel", params)
        assert df is not None
        assert results["qmax_AOF"] > 0

    def test_composite_with_test_point(self):
        """Generate Composite IPR from test point."""
        params = {"Pr": 3000, "Pb": 2000, "qtest": 800, "Pwf_test": 1500, "N": 50}
        df, results, warnings = generate_ipr_curve("Composite", params)
        assert df is not None
        assert "qb" in results

    def test_standing_with_test_point(self):
        """Generate Standing IPR from test point."""
        params = {
            "Pr": 3000, "Pb": 2000, "qtest": 800, "Pwf_test": 1500,
            "fe_old": 0.8, "fe_new": 1.0, "N": 50,
        }
        df, results, warnings = generate_ipr_curve("Standing", params)
        assert df is not None
        assert "FE" in results
        assert results["qmax_AOF"] > 0

    def test_fetkovich_with_test_point(self):
        """Generate Fetkovich IPR from test point via dispatcher."""
        params = {
            "Pr": 3000, "Pb": 2000, "qtest": 800, "Pwf_test": 1500,
            "user_n": 0.85, "N": 50,
        }
        df, results, warnings = generate_ipr_curve("Fetkovich", params)
        assert df is not None
        assert results["qmax_AOF"] > 0
        assert "C" in results
        assert "n" in results
        assert "J" in results

    def test_missing_pr_returns_error(self):
        """Missing Pr should return error."""
        params = {"qtest": 800, "Pwf_test": 1500, "N": 50}
        df, results, warnings = generate_ipr_curve("Vogel", params)
        assert df is None
        assert len(warnings) > 0

    def test_unknown_method(self):
        """Unknown method name should return error."""
        params = {"Pr": 3000, "N": 50}
        df, results, warnings = generate_ipr_curve("UnknownMethod", params)
        assert df is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
