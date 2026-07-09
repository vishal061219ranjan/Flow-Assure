"""
FLOW ASSURE: Oil & Gas Well Nodal Analysis Tool
"""

from __future__ import annotations

import io
import math
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

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
from backend.plots import plot_comparison, plot_ipr_curve, plot_sensitivity, plot_vlp_sensitivity, plot_nodal_analysis
from backend.pvt import calculate_all_pvt
from backend.validation import check_ipr_suitability, validate_inputs
from backend.vlp import generate_vlp_curve, find_operating_point


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

im = Image.open("Logo.png")

st.set_page_config(
    page_title="Flow Assure",
    page_icon=im,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* 1. Target ALL buttons globally (including the + and - inside Number Inputs) */
    button, 
    div[data-testid="stNumberInput"] button {
        cursor: pointer !important;
    }
    
    /* 2. Target the Expander headers (Reservoir Data, Model Options, etc.) */
    summary,
    div[data-testid="stExpander"] summary {
        cursor: pointer !important;
    }
    
    /* 3. Target the Selectbox dropdowns */
    div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div {
        cursor: pointer !important;
    }
    /* 4. Compact Sidebar: smaller text, less padding */
    section[data-testid="stSidebar"] {
        width: 380px !important;
    }
    
    /* Make text smaller */
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label p,
    section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {
        font-size: 13px !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        font-size: 18px !important;
        margin-top: 5px !important;
        margin-bottom: 5px !important;
    }
    
    section[data-testid="stSidebar"] h3 {
        font-size: 15px !important;
        margin-top: 5px !important;
        margin-bottom: 5px !important;
    }

    /* Reduce default large gaps safely without causing overlap */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stForm"] > div,
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem !important;
    }
    /* ── Flow Assure Custom UI Styling ───────────────────────────── */
    
    .main-header {
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.88);
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
        min-height: 125px;
        margin-bottom: 0.75rem;
    }
    
    .metric-card .value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    
    .metric-card .label {
        margin-top: 0.45rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }
    
    .metric-card .source {
        margin-top: 0.45rem;
        font-size: 0.74rem;
        color: #64748b;
    }
    
    .warning-box {
        padding: 0.85rem 1rem;
        margin: 0.55rem 0;
        border-radius: 12px;
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        color: #78350f;
        font-size: 0.92rem;
        font-weight: 500;
    }
    
    .error-box {
        padding: 0.85rem 1rem;
        margin: 0.55rem 0;
        border-radius: 12px;
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        color: #7f1d1d;
        font-size: 0.92rem;
        font-weight: 500;
    }
    
    .info-box {
        padding: 0.9rem 1rem;
        margin: 0.65rem 0;
        border-radius: 12px;
        background: #eff6ff;
        border-left: 5px solid #3b82f6;
        color: #1e3a8a;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    
    /* 5. Hide edit and GitHub buttons from deployed app */
    .stDeployButton, 
    button[title="Edit this app"],
    a[title="View app source on GitHub"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>
        <span style='color: #FFFFFF;'>Flow</span> <span style='color: #1089A5;'>Assure</span>
    </h1>
    <p>Generate complete oil & gas well nodal analysis (IPR and VLP) using petroleum engineering correlations.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUT PANEL
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Input Panel")

    # ── Model Selection ───────────────────────────────────────────────────
    st.markdown("### Model Options")

    ipr_methods = [
        "Darcy (Productivity Index)",
        "Vogel",
        "Composite (Darcy + Vogel)",
        "Standing (Flow Efficiency)",
        "Fetkovich",
    ]
    ipr_method = st.selectbox(
        "IPR Method",
        ipr_methods,
        index=2,  # default: Composite
        help="Select the inflow performance relationship model.",
    )

    pvt_methods = ["Standing", "Vasquez-Beggs"]
    pvt_method = st.selectbox(
        "PVT Correlation",
        pvt_methods,
        index=0,
        help="Correlation used for oil PVT property estimation.",
    )

    # Map display names to internal keys
    IPR_KEY_MAP = {
        "Darcy (Productivity Index)": "Darcy",
        "Vogel": "Vogel",
        "Composite (Darcy + Vogel)": "Composite",
        "Standing (Flow Efficiency)": "Standing",
        "Fetkovich": "Fetkovich",
    }
    ipr_key = IPR_KEY_MAP[ipr_method]

    st.markdown("---")

    # ── Reservoir Data ────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### Reservoir Data")

        Pr = st.number_input("Reservoir Pressure, Pr (psia)", value=3000.0, min_value=0.0, step=100.0, format="%.1f")
        Pb_input = st.number_input("Bubble Point Pressure, Pb (psia)", value=2000.0, min_value=0.0, step=100.0, format="%.1f", help="Set to 0 to auto-calculate from GOR.")
        T = st.number_input("Reservoir Temperature, T (°F)", value=180.0, min_value=0.0, step=10.0, format="%.1f")

        reservoir_type = st.selectbox(
            "Reservoir Type",
            ["Undersaturated Oil", "Saturated Oil", "Gas Well"],
            index=0,
        )

        st.markdown("**Rock Properties**")
        skin = st.number_input("Skin Factor, s", value=0.0, step=0.5, format="%.2f")
        
        use_reservoir_props = st.checkbox("Calculate J from Reservoir Properties", value=True)
        if use_reservoir_props:
            k = st.number_input("Permeability, k (md)", value=50.0, min_value=0.0, step=5.0, format="%.2f")
            h = st.number_input("Net Pay Thickness, h (ft)", value=40.0, min_value=0.0, step=5.0, format="%.2f")
            re = st.number_input("Drainage Radius, re (ft)", value=1000.0, min_value=1.0, step=100.0, format="%.2f")
            rw = st.number_input("Wellbore Radius, rw (ft)", value=0.328, min_value=0.01, step=0.05, format="%.3f")
        else:
            # Set to 0.0 so that the backend knows not to use them for J calculation
            k, h, re, rw = 0.0, 0.0, 0.0, 0.0

    st.markdown("---")

    # ── Fluid Data ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### Fluid Data")

        api = st.number_input("Oil API Gravity (°API)", value=35.0, min_value=5.0, max_value=70.0, step=1.0, format="%.1f")
        gas_sg = st.number_input("Gas Specific Gravity (air=1)", value=0.75, min_value=0.3, max_value=2.0, step=0.05, format="%.3f")

        qtest = st.number_input("Oil Test Rate, qtest (STB/day)", value=800.0, min_value=0.0, step=50.0, format="%.1f")
        Pwf_test = st.number_input("Pwf at Test Point (psia)", value=1500.0, min_value=0.0, step=100.0, format="%.1f")

        st.markdown("**Additional Fluid Properties**")
        GOR = st.number_input("Gas-Oil Ratio, GOR (scf/STB)", value=500.0, min_value=0.0, step=50.0, format="%.2f", help="Producing GOR; used as proxy for Rs if Pb is auto-calculated.")
        water_cut = st.number_input("Water Cut (fraction)", value=0.0, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        water_sg = st.number_input("Water Specific Gravity", value=1.0, min_value=0.9, max_value=1.2, step=0.01, format="%.2f")
        
        if pvt_method == "Vasquez-Beggs":
            Psep = st.number_input("Separator Pressure (psig)", value=100.0, min_value=0.0, step=25.0, format="%.2f")
            Tsep = st.number_input("Separator Temperature (°F)", value=75.0, min_value=0.0, step=5.0, format="%.2f")
        else:
            Psep, Tsep = 100.0, 75.0
            
        salinity = st.number_input("Salinity (wt% NaCl)", value=0.0, min_value=0.0, max_value=30.0, step=0.5, format="%.2f")

    st.markdown("---")

    # ── Manual PVT Overrides ──────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### Manual PVT Overrides")
        st.caption("Leave at 0 to auto-calculate.")

        Bo_manual = st.number_input("Bo (RB/STB) — manual", value=0.0, min_value=0.0, step=0.01, format="%.4f")
        Rs_manual = st.number_input("Rs (scf/STB) — manual", value=0.0, min_value=0.0, step=10.0, format="%.2f")
        mu_o_manual = st.number_input("μo (cP) — manual", value=0.0, min_value=0.0, step=0.1, format="%.3f")

    st.markdown("---")

    # ── Method-Specific Parameters ────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### Method Parameters")

    fe_old_standing = None
    fe_new_standing = 1.0
    fetk_user_n = None
    fetk_q_test2 = None
    fetk_pwf_test2 = None
    J_manual = 0.0
    darcy_j_source = "test"

    if ipr_key == "Standing":
        fe_source = st.radio(
            "FE Source",
            ["Provide FE directly", "Calculate from Skin & Radius"],
            index=0,
            help="How to determine the flow efficiency at the time of the test.",
        )
        if fe_source == "Provide FE directly":
            fe_old_standing = st.number_input(
                "FE at Test Conditions (FE_old)", value=1.0,
                min_value=0.1, max_value=3.0, step=0.1, format="%.2f",
                help="FE at the time the well test was conducted."
            )
        else:
            st.caption("FE will be calculated from re, rw, and skin in the Reservoir Rock Properties section above.")

        fe_new_standing = st.number_input(
            "Desired New FE (FE_new)", value=1.0,
            min_value=0.1, max_value=3.0, step=0.1, format="%.2f",
            help="FE for the generated IPR curve. 1.0 = ideal (zero-skin)."
        )
    elif ipr_key == "Fetkovich":
        fetk_n_source = st.radio(
            "Exponent (n) Source",
            ["Default (n=1.0, Darcy flow)", "Provide n manually", "Calculate from 2nd test point"],
            index=0,
            help="How to determine the Fetkovich deliverability exponent.",
        )

        fetk_user_n = None
        fetk_q_test2 = None
        fetk_pwf_test2 = None

        if fetk_n_source == "Provide n manually":
            fetk_user_n = st.number_input(
                "Fetkovich Exponent, n", value=1.0,
                min_value=0.5, max_value=1.0, step=0.05, format="%.2f",
                help="n=1.0 (Darcy flow) to n=0.5 (turbulent flow)"
            )
        elif fetk_n_source == "Calculate from 2nd test point":
            st.caption("Provide a second multi-rate test point to solve for n analytically.")
            fetk_q_test2 = st.number_input(
                "2nd Test Rate, q_test2 (STB/day)", value=0.0,
                min_value=0.0, step=50.0, format="%.2f",
                help="Oil rate at the second test point."
            )
            fetk_pwf_test2 = st.number_input(
                "2nd Test Pwf, pwf_test2 (psia)", value=0.0,
                min_value=0.0, step=100.0, format="%.2f",
                help="Flowing BHP at the second test point."
            )
            if fetk_q_test2 <= 0:
                fetk_q_test2 = None
            if fetk_pwf_test2 <= 0:
                fetk_pwf_test2 = None
    elif ipr_key == "Darcy":
        darcy_j_source_label = st.radio(
            "Select Darcy J Source",
            [
                "Use test point",
                "Use reservoir properties",
                "Enter J manually",
            ],
            index=1 if use_reservoir_props else 0,
            help="Choose how Darcy productivity index J should be calculated.",
        )

        if darcy_j_source_label == "Use test point":
            darcy_j_source = "test"

        elif darcy_j_source_label == "Use reservoir properties":
            darcy_j_source = "reservoir"

        else:
            darcy_j_source = "manual"
            J_manual = st.number_input(
                "Productivity Index, J (STB/day/psi)",
                value=0.0,
                min_value=0.0,
                step=0.1,
                format="%.3f",
                help="Enter J directly.",
            )

    N_points = st.slider("Number of Curve Points", min_value=20, max_value=200,
                         value=50, step=10)

    st.markdown("---")

    # ── Comparison & Sensitivity ──────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### Advanced Options")

    comparison_mode = st.checkbox("Compare All IPR Methods", value=False,
                                 help="Overlay all applicable IPR curves on one plot.")

    sensitivity_mode = st.checkbox("Sensitivity Analysis", value=False)
    sensitivity_param = None
    sensitivity_values = []
    if sensitivity_mode:
        sensitivity_param = st.selectbox(
            "Parameter to Vary",
            ["Reservoir Pressure (Pr)", "Bubble Point (Pb)", "Skin Factor (s)",
             "Flow Efficiency (FE)", "Fetkovich n"],
        )
        sens_min = st.number_input("Min Value", value=0.0, format="%.2f")
        sens_max = st.number_input("Max Value", value=5000.0, format="%.2f")
        sens_steps = st.slider("Number of Steps", 3, 10, 5)
        if sens_min < sens_max:
            sensitivity_values = np.linspace(sens_min, sens_max, sens_steps).tolist()

    st.markdown("---")

    # ── Well Configuration (VLP) ──────────────────────────────────────────
    with st.container(border=True):
        st.markdown("### Pipe & Wellbore Data (VLP)")

        enable_vlp = st.toggle(
            "Enable VLP / Nodal Analysis", value=True,
            help="Calculate VLP curve and detailed nodal analysis.",
        )

        vlp_Pwh = 200.0
        vlp_TVD = 5000.0
        vlp_MD = 5000.0
        vlp_d_tubing = 2.441
        vlp_roughness = 0.0006
        vlp_T_surface = 100.0
        vlp_segments = 50
        vlp_q_target = 1000.0
        use_polynomial = False
        apply_bo_undersat_correction = False
        friction_method = "Churchill"

        vlp_correlation = "Hagedorn-Brown"
        vlp_label = "Modified Hagedorn-Brown"
        bb_inclination_deg = 90.0
        bb_flow_direction = "Uphill / Production"
        bb_apply_payne = True
        vlp_sensitivity_mode = False
        vlp_sensitivity_param = None
        vlp_sensitivity_values = []

        if enable_vlp:
            vlp_correlation_label = st.selectbox(
                "VLP Correlation",
                ["Modified Hagedorn-Brown", "Beggs-Brill"],
                index=0,
                help="Select the VLP pressure-gradient correlation.",
            )

            if vlp_correlation_label == "Beggs-Brill":
                vlp_correlation = "Beggs-Brill"
                vlp_label = "Beggs-Brill"
            else:
                vlp_correlation = "Hagedorn-Brown"
                vlp_label = "Modified Hagedorn-Brown"

            st.markdown(f"**Inputs for {vlp_label}**")

            # Common VLP inputs for both correlations
            vlp_Pwh = st.number_input("Wellhead Pressure, Pwh (psia)", value=200.0, min_value=14.7, step=25.0, format="%.2f")
            vlp_d_tubing = st.number_input("Tubing ID (inches)", value=2.441, min_value=0.5, max_value=12.0, step=0.1, format="%.3f")
            vlp_T_surface = st.number_input(
                "Surface Temperature (°F)",
                value=100.0,
                min_value=32.0,
                step=5.0,
                format="%.1f",
            )
            vlp_segments = st.number_input(
                "Number of Segments",
                value=50,
                min_value=10,
                max_value=500,
                step=10,
            )
            apply_bo_undersat_correction = st.checkbox("Apply undersaturated Bo correction in VLP",value=False,help=("OFF keeps the current VLP behavior unchanged. "
                                                                                                                     "ON applies Bo(P)=Bo(Pb)*exp[-co(P-Pb)] when local pressure is above Pb."
                                                                                                                     ),
            )

            # Hagedorn-Brown-only inputs
            if vlp_correlation == "Hagedorn-Brown":
                vlp_TVD = st.number_input("True Vertical Depth, TVD (ft)", value=5000.0, min_value=100.0, step=100.0, format="%.1f")
                vlp_MD = vlp_TVD  # Defaults for Beggs-Brill-only variables
                bb_inclination_deg = 90.0
                bb_flow_direction = "Uphill / Production"
                bb_apply_payne = True

                vlp_roughness = st.number_input(
                    "Tubing Roughness, ε (inches)",
                    value=0.0006,
                    min_value=0.0,
                    max_value=0.01,
                    step=0.0001,
                    format="%.4f",
                )

                use_polynomial = st.toggle(
                    "Use H-B Empirical Correlation (Instead of Digitized Chart 3)",
                    value=False,
                )

                friction_method_label = st.selectbox(
                    "Friction Factor Correlation",
                    ["Churchill (1977)", "Chen (1979)"],
                    index=0,
                    help="Churchill: continuous across all regimes. Chen: explicit Colebrook-White approximation.",
                )

                friction_method = "Chen" if "Chen" in friction_method_label else "Churchill"

            # Beggs-Brill-only inputs
            else:
                # H-B-only variables disabled for Beggs-Brill
                vlp_roughness = 0.0
                use_polynomial = False

                bb_geometry_mode = st.radio(
                    "Parameter to Calculate",
                    ["Inclination Angle (α)", "Measured Depth (MD)", "True Vertical Depth (TVD)"],
                    index=0,
                    help="Select the parameter to be calculated. The other two will be provided as inputs."
                )

                if bb_geometry_mode == "Inclination Angle (α)":
                    col1, col2 = st.columns(2)
                    with col1:
                        vlp_TVD = st.number_input("True Vertical Depth, TVD (ft)", value=5000.0, min_value=100.0, step=100.0, format="%.1f")
                    with col2:
                        vlp_MD = st.number_input("Measured Depth, MD (ft)", value=vlp_TVD, min_value=vlp_TVD, step=100.0, format="%.1f", help="MD must be ≥ TVD.")
                    
                    bb_inclination_deg = math.degrees(math.asin(min(max(vlp_TVD / max(vlp_MD, 1e-9), 0.0), 1.0)))
                    st.number_input("Calculated Inclination, α (deg)", value=float(bb_inclination_deg), disabled=True, help="Automatically calculated: sin(α) = TVD / MD.")
                
                elif bb_geometry_mode == "Measured Depth (MD)":
                    col1, col2 = st.columns(2)
                    with col1:
                        vlp_TVD = st.number_input("True Vertical Depth, TVD (ft)", value=5000.0, min_value=100.0, step=100.0, format="%.1f")
                    with col2:
                        bb_inclination_deg = st.number_input("Inclination Angle, α (deg)", value=90.0, min_value=0.1, max_value=90.0, step=5.0, format="%.1f", help="90° = vertical upward.")
                    
                    vlp_MD = vlp_TVD / math.sin(math.radians(bb_inclination_deg))
                    st.number_input("Calculated Measured Depth, MD (ft)", value=float(vlp_MD), disabled=True, help="Automatically calculated: MD = TVD / sin(α).")
                
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        vlp_MD = st.number_input("Measured Depth, MD (ft)", value=5000.0, min_value=100.0, step=100.0, format="%.1f")
                    with col2:
                        bb_inclination_deg = st.number_input("Inclination Angle, α (deg)", value=90.0, min_value=0.1, max_value=90.0, step=5.0, format="%.1f", help="90° = vertical upward.")
                    
                    vlp_TVD = vlp_MD * math.sin(math.radians(bb_inclination_deg))
                    st.number_input("Calculated True Vertical Depth, TVD (ft)", value=float(vlp_TVD), disabled=True, help="Automatically calculated: TVD = MD * sin(α).")

                bb_flow_direction = "Uphill / Production"

                allow_bb_downhill = st.checkbox(
                    "Show downhill/injection option — advanced signed-angle case",
                    value=False,
                    help=(
                        "Keep this OFF for normal production nodal analysis. "
                        "Downhill/Injection uses negative inclination, so the elevation gradient "
                        "can become negative."
                    ),
                )

                if allow_bb_downhill:
                    bb_flow_direction = st.selectbox(
                        "Flow Direction (signed inclination)",
                        ["Uphill / Production", "Downhill / Injection"],
                        index=0,
                        help=(
                            "Uphill/Production uses positive inclination. "
                            "Downhill/Injection uses negative inclination and is a special case."
                        ),
                    )

                    if bb_flow_direction == "Downhill / Injection":
                        st.warning(
                            "Downhill / Injection is a signed-angle special case. "
                            "It can make the hydrostatic/elevation gradient negative. "
                            "Do not use this for normal production VLP/nodal analysis."
                        )
                else:
                    st.caption(
                        "Flow direction fixed as Uphill / Production for normal production nodal analysis."
                    )

                bb_apply_payne = st.checkbox(
                    "Apply Payne Correction",
                    value=True,
                    help="Applies Beggs-Brill Payne correction to liquid holdup.",
                )

                friction_method_label = st.selectbox(
                    "Beggs-Brill Friction Factor",
                    ["Churchill Smooth Pipe", "Beggs-Brill Smooth Pipe"],
                    index=0,
                )

                friction_method = (
                    "Beggs-Brill Smooth"
                    if "Beggs-Brill" in friction_method_label
                    else "Churchill"
                )

            # ── VLP Sensitivity Analysis ───────────────────────────────────
            st.markdown("---")
            vlp_sensitivity_mode = st.checkbox("VLP Sensitivity Analysis", value=False,
                                               help="Generate multiple VLP curves by varying GOR or Tubing ID.")
            vlp_sensitivity_param = None
            vlp_sensitivity_values = []
            if vlp_sensitivity_mode:
                vlp_sensitivity_param = st.selectbox(
                    "VLP Parameter to Vary",
                    ["GOR (scf/STB)", "Tubing ID (inches)"],
                    index=0,
                )
                if "GOR" in vlp_sensitivity_param:
                    vlp_sens_min = st.number_input("Min GOR", value=200.0, min_value=0.0, step=50.0, format="%.0f")
                    vlp_sens_max = st.number_input("Max GOR", value=1500.0, min_value=0.0, step=50.0, format="%.0f")
                else:
                    vlp_sens_min = st.number_input("Min Tubing ID (in)", value=1.5, min_value=0.5, step=0.25, format="%.3f")
                    vlp_sens_max = st.number_input("Max Tubing ID (in)", value=4.0, min_value=0.5, step=0.25, format="%.3f")
                vlp_sens_steps = st.slider("Number of VLP Sensitivity Steps", 2, 10, 5)
                if vlp_sens_min < vlp_sens_max:
                    vlp_sensitivity_values = np.linspace(vlp_sens_min, vlp_sens_max, vlp_sens_steps).tolist()

    st.markdown("---")

    # Generate button
    generate = st.button("Generate", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Build the inputs dict
inputs = {
    "Pr": Pr,
    "Pb_manual": Pb_input if Pb_input > 0 else None,
    "T": T,
    "api": api,
    "gas_sg": gas_sg,
    "qtest": qtest if qtest > 0 else None,
    "Pwf_test": Pwf_test if Pwf_test > 0 else None,
    "GOR": GOR if GOR > 0 else None,
    "water_cut": water_cut,
    "k": k if k > 0 else None,
    "h": h if h > 0 else None,
    "re": re if re > 0 else None,
    "rw": rw if rw > 0 else None,
    "s": skin,
    "Psep": Psep,
    "Tsep": Tsep,
    "salinity": salinity,
    "Bo_manual": Bo_manual if Bo_manual > 0 else None,
    "Rs_manual": Rs_manual if Rs_manual > 0 else None,
    "mu_o_manual": mu_o_manual if mu_o_manual > 0 else None,
    "fe_old_standing": fe_old_standing,
    "fe_new_standing": fe_new_standing,
    "fetk_user_n": fetk_user_n if ipr_key == "Fetkovich" else None,
    "fetk_q_test2": fetk_q_test2 if ipr_key == "Fetkovich" else None,
    "fetk_pwf_test2": fetk_pwf_test2 if ipr_key == "Fetkovich" else None,
    "J_manual": J_manual if J_manual > 0 else None,
    "N": N_points,
    "reservoir_type": reservoir_type,
}

# Always compute (for the result display even before button click)
if generate or True:  # Always show results when inputs change
    # ── Validate Inputs ───────────────────────────────────────────────────
    errors, val_warnings = validate_inputs(inputs)

    # ── Run PVT ───────────────────────────────────────────────────────────
    pvt_output = calculate_all_pvt(inputs, pvt_method)
    pvt_results = pvt_output["results"]
    pvt_values = pvt_output["values"]
    pvt_warnings = pvt_output["warnings"]

    # Get computed Pb for IPR use
    Pb_computed = pvt_values.get("Pb") or (Pb_input if Pb_input > 0 else None)
    inputs["Pb_calculated"] = Pb_computed

    # Check IPR suitability
    suitability_warnings = check_ipr_suitability(ipr_key, inputs)

    # ── Build IPR Parameters ──────────────────────────────────────────────
    ipr_params = {
        "Pr": Pr,
        "Pb": Pb_computed,
        "qtest": qtest if qtest > 0 else None,
        "Pwf_test": Pwf_test if Pwf_test > 0 else None,
        "J": J_manual if J_manual > 0 else None,
        "k": k if k > 0 else None,
        "h": h if h > 0 else None,
        "mu_o": pvt_values.get("mu_o"),
        "Bo": pvt_values.get("Bo"),
        "re": re,
        "rw": rw,
        "s": skin,
        "darcy_j_source": darcy_j_source,#------------------NEW
        # Standing FE params
        "fe_old": fe_old_standing if fe_old_standing and fe_old_standing > 0 else None,
        "fe_new": fe_new_standing,
        # Fetkovich params (new module)
        "user_n": fetk_user_n if ipr_key == "Fetkovich" else None,
        "q_test2": fetk_q_test2 if ipr_key == "Fetkovich" else None,
        "pwf_test2": fetk_pwf_test2 if ipr_key == "Fetkovich" else None,
        "N": N_points,
    }

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_pvt, tab_ipr, tab_vlp_curve, tab_nodal, tab_ipr_table, tab_vlp_table, tab_assumptions = st.tabs([
        "PVT Results",
        "IPR Curve",
        "VLP Curve",
        "Nodal Analysis",
        "IPR Calculation",
        "VLP Calculation",
        "Assumptions & Info",
    ])

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1: PVT RESULTS
    # ══════════════════════════════════════════════════════════════════════
    with tab_pvt:
        st.markdown("#### Calculated PVT Properties")
        st.caption(f"Correlation: **{pvt_method}** | Beggs-Robinson (viscosity) | DAK (Z-factor) | Lee-Gonzalez-Eakin (gas viscosity)")

        # Show errors
        for err in errors:
            st.markdown(f'<div class="error-box">❌ {err}</div>', unsafe_allow_html=True)

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        def _metric_card(col, result):
            """Render a styled metric card."""
            if result is None:
                return
            val_str = f"{result.value:.4f}" if result.value is not None else "N/A"
            col.markdown(f"""
            <div class="metric-card">
                <div class="value">{val_str}</div>
                <div class="label">{result.name} ({result.symbol}), {result.unit}</div>
                <div class="source">{result.source}</div>
            </div>
            """, unsafe_allow_html=True)

        _metric_card(col1, pvt_results.get("Pb"))
        _metric_card(col2, pvt_results.get("Rs"))
        _metric_card(col3, pvt_results.get("Bo"))
        _metric_card(col4, pvt_results.get("mu_o"))

        st.markdown("")
        col5, col6, col7, col8 = st.columns(4)
        _metric_card(col5, pvt_results.get("Z"))
        _metric_card(col6, pvt_results.get("Bg"))
        _metric_card(col7, pvt_results.get("mu_g"))
        _metric_card(col8, pvt_results.get("Bw"))

        # PVT Results Table
        st.markdown("---")
        st.markdown("#### Detailed PVT Results")
        pvt_table_data = [r.to_dict() for r in pvt_results.values()]
        pvt_df = pd.DataFrame(pvt_table_data)
        st.dataframe(pvt_df, use_container_width=True, hide_index=True)

        # Warnings
        if pvt_warnings:
            st.markdown("#### ⚠️ PVT Warnings")
            for w in pvt_warnings:
                st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2: IPR CURVE
    # ══════════════════════════════════════════════════════════════════════
    with tab_ipr:
        # Show suitability warnings
        for w in suitability_warnings:
            st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)

        if errors:
            st.error("Fix input errors before generating IPR curves. Check the PVT Results tab.")
        else:
            # ── Single Curve Mode ─────────────────────────────────────────
            if not comparison_mode and not sensitivity_mode:
                df_ipr, key_results, ipr_warnings = generate_ipr_curve(ipr_key, ipr_params)

                if df_ipr is not None:
                    # Test point
                    op_point = None
                    if( qtest > 0 and Pwf_test > 0 and (ipr_key != "Darcy" or darcy_j_source == "test")
                    ):
                        op_point = (qtest, Pwf_test)

                    # Bubble point (for Composite)
                    bp_point = None
                    if ipr_key == "Composite" and "qb" in key_results and Pb_computed:
                        bp_point = (key_results["qb"], Pb_computed)

                    fig = plot_ipr_curve(
                        df_ipr,
                        key_results.get("method", ipr_method),
                        operating_point=op_point,
                        bubble_point=bp_point,
                        key_results=key_results,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

                    # Key results
                    st.markdown("#### Key Results")

                    
                    # 1. Initialize empty lists to build our table columns dynamically
                    vals = []
                    params = []

                    # 2. qmax (Always present)
                    qmax_val = key_results.get('qomax', key_results.get('qmax_AOF', key_results.get('qmax_STBd', 'N/A')))
                    if isinstance(qmax_val, (int, float)):
                        qmax_val = f"{qmax_val:.2f}"
                    params.append("Absolute Open Flow (qmax)")
                    vals.append(f"{qmax_val} STB/day")

                    # 3. Add J if it exists
                    if "J" in key_results:
                        params.append("Productivity Index (J)")
                        vals.append(f"{key_results['J']} STB/day/psi")
                    elif "J_new_STBd_psi" in key_results:
                        params.append("Productivity Index (J)")
                        vals.append(f"{key_results['J_new_STBd_psi']:.4f} STB/day/psi")

                    # 4. Add qb if it exists
                    if "qb" in key_results:
                        params.append("Bubble Point Rate (qb)")
                        vals.append(f"{key_results['qb']} STB/day")
                    elif "q_at_Pb_STBd" in key_results:
                        params.append("Bubble Point Rate (qb)")
                        vals.append(f"{key_results['q_at_Pb_STBd']:.2f} STB/day")

                    # 5. Add Gas Deliverability variables if they exist
                    if "C" in key_results:
                        params.append("Deliverability (C)")
                        vals.append(f"{key_results['C']:.6f}")
                        params.append("Exponent (n)")
                        vals.append(f"{key_results['n']}")

                    # 6. Add Flow Efficiency (FE) variables
                    if "FE_new" in key_results:
                        params.append("Flow Efficiency (New)")
                        vals.append(f"{key_results['FE_new']}")
                    elif "FE" in key_results:
                        params.append("Flow Efficiency")
                        vals.append(f"{key_results['FE']}")
    
                    if "FE_test" in key_results:
                        params.append("Flow Efficiency (Test)")
                        vals.append(f"{key_results['FE_test']}")

                    # 7. Add Well Condition / Sub-case
                    if "well_condition" in key_results:
                        params.append("Well Condition")
                        vals.append(f"{key_results['well_condition']}")
                    elif "sub_case" in key_results:
                        params.append("Well Condition (Sub-case)")
                        vals.append(f"{key_results['sub_case']}")
    
                    # 8. Add Reservoir Type
                    if "reservoir_type" in key_results:
                        params.append("Reservoir Type")
                        vals.append(f"{key_results['reservoir_type']}")

                    # 9. Add Method Name
                    params.append("IPR Method")
                    vals.append(f"{key_results.get('method', 'Selected IPR Method')}")

                    # 10. Convert the dynamic lists into a Pandas DataFrame and display it
                    df_results = pd.DataFrame({
                        "Parameter": params,
                        "Value": vals
                    })
                    st.dataframe(df_results, hide_index=True, use_container_width=True)

                    # 11. Keep the helpful warning boxes at the bottom
                    if key_results.get("FE_limit_applied"):
                        st.markdown(
                            f'<div class="warning-box">FE &gt; 1 pressure limit applied. '
                            f'Curve terminates at Pwf = '
                            f'{key_results.get("Pwf_min_valid_psi", 0):.1f} psia.</div>',
                            unsafe_allow_html=True,
                        )

                    # Warnings
                    for w in ipr_warnings:
                        st.markdown(f'<div class="warning-box">{w}</div>',
                                    unsafe_allow_html=True)

                    # Downloads
                    st.markdown("---")
                    st.markdown("#### Downloads")
                    dl_col1, dl_col2, dl_col3 = st.columns(3)

                    # CSV download
                    csv_data = df_ipr.to_csv(index=False)
                    dl_col1.download_button(
                        "Download IPR Data (CSV)",
                        csv_data,
                        "ipr_data.csv",
                        "text/csv",
                        use_container_width=True,
                    )

                    # Plot image download
                    img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
                    dl_col2.download_button(
                        "Download Plot (PNG)",
                        img_bytes,
                        "ipr_curve.png",
                        "image/png",
                        use_container_width=True,
                    )

                    # Summary text
                    summary_lines = [
                        "Flow Assure — Calculation Summary",
                        "=" * 40,
                        f"IPR Method: {key_results.get('method', ipr_method)}",
                        f"PVT Correlation: {pvt_method}",
                        "",
                        "Input Parameters:",
                        f"  Pr = {Pr} psia",
                        f"  Pb = {Pb_computed} psia",
                        f"  T = {T} °F",
                        f"  API = {api} °API",
                        f"  Gas SG = {gas_sg}",
                        f"  qtest = {qtest} STB/day",
                        f"  Pwf_test = {Pwf_test} psia",
                        "",
                        "PVT Results:",
                    ]
                    for r in pvt_results.values():
                        summary_lines.append(f"  {r}")
                    summary_lines.extend([
                        "",
                        "IPR Results:",
                        f"  qmax/AOF = {key_results.get('qmax_AOF', 'N/A')} STB/day",
                    ])
                    if "J" in key_results:
                        summary_lines.append(f"  J = {key_results['J']} STB/day/psi")
                    if "qb" in key_results:
                        summary_lines.append(f"  qb = {key_results['qb']} STB/day")

                    summary_text = "\n".join(summary_lines)
                    dl_col3.download_button(
                        "Download Summary (TXT)",
                        summary_text,
                        "calculation_summary.txt",
                        "text/plain",
                        use_container_width=True,
                    )
                else:
                    for w in ipr_warnings:
                        st.markdown(f'<div class="error-box">{w}</div>',
                                    unsafe_allow_html=True)

            # ── Comparison Mode ───────────────────────────────────────────
            elif comparison_mode and not sensitivity_mode:
                st.markdown("#### IPR Method Comparison")

                curves_dict = {}
                all_results = {}

                method_keys = ["Darcy", "Vogel", "Composite", "Standing", "Fetkovich"]
                for mk in method_keys:
                    params_copy = dict(ipr_params)
                    if mk == "Standing":
                        params_copy["fe_old"] = fe_old_standing if fe_old_standing and fe_old_standing > 0 else None
                        params_copy["fe_new"] = fe_new_standing
                    if mk == "Fetkovich":
                        # For comparison mode, use defaults (single test, n=1)
                        params_copy.pop("user_n", None)
                        params_copy.pop("q_test2", None)
                        params_copy.pop("pwf_test2", None)

                    df_c, kr_c, w_c = generate_ipr_curve(mk, params_copy)
                    if df_c is not None:
                        label = kr_c.get("method", mk)
                        curves_dict[label] = df_c
                        all_results[label] = kr_c

                if curves_dict:
                    op_point = None
                    if( qtest > 0 and Pwf_test > 0 and ("Darcy" not in method_keys or darcy_j_source == "test")
                    ):
                        op_point = (qtest, Pwf_test)
                    fig_comp = plot_comparison(curves_dict, operating_point=op_point)
                    st.plotly_chart(fig_comp, use_container_width=True)

                    # Summary table
                    comp_data = []
                    for label, kr in all_results.items():
                        comp_data.append({
                            "Method": label,
                            "qmax/AOF (STB/day)": kr.get("qmax_AOF", "N/A"),
                            "J (STB/day/psi)": kr.get("J", "—"),
                        })
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("Could not generate any IPR curves. Check inputs.")

            # ── Sensitivity Mode ──────────────────────────────────────────
            elif sensitivity_mode and sensitivity_values:
                st.markdown(f"#### Sensitivity: {sensitivity_param}")

                curves_list = []
                for val in sensitivity_values:
                    params_copy = dict(ipr_params)

                    if "Reservoir Pressure" in sensitivity_param:
                        params_copy["Pr"] = val
                        label = f"Pr = {val:.0f}"
                    elif "Bubble Point" in sensitivity_param:
                        params_copy["Pb"] = val
                        label = f"Pb = {val:.0f}"
                    elif "Skin" in sensitivity_param:
                        params_copy["s"] = val
                        # Recalculate J with new skin
                        if k and h and pvt_values.get("mu_o") and pvt_values.get("Bo"):
                            j_new, _ = calculate_J_from_reservoir(
                                k, h, pvt_values["mu_o"], pvt_values["Bo"], re, rw, val)
                            params_copy["J"] = j_new
                        label = f"s = {val:.1f}"
                    elif "Flow Efficiency" in sensitivity_param:
                        params_copy["fe_new"] = val
                        label = f"FE = {val:.2f}"
                    elif "Fetkovich" in sensitivity_param:
                        params_copy["user_n"] = max(0.5, min(val, 1.0))
                        label = f"n = {val:.2f}"
                    else:
                        label = f"{val:.1f}"

                    df_s, kr_s, w_s = generate_ipr_curve(ipr_key, params_copy)
                    if df_s is not None:
                        curves_list.append((label, df_s))

                if curves_list:
                    fig_sens = plot_sensitivity(curves_list, sensitivity_param, ipr_method)
                    st.plotly_chart(fig_sens, use_container_width=True)
                else:
                    st.warning("Could not generate sensitivity curves.")

    # ══════════════════════════════════════════════════════════════════════
    #  VLP CURVE & NODAL ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    
    # Pre-generate VLP for both tabs if enabled
    df_vlp, vlp_kr, vlp_w = None, None, None
    df_ipr_n, kr_n, w_n = None, None, None
    df_vlp_plot = None
    op = None
    if enable_vlp and not errors:
        df_ipr_n, kr_n, w_n = generate_ipr_curve(ipr_key, ipr_params)
        if df_ipr_n is not None:
            qmax_n = kr_n.get("qmax_AOF", kr_n.get("qomax", kr_n.get("qmax_STBd", 3000)))
            if isinstance(qmax_n, str): qmax_n = 3000.0
            vlp_qmax = float(qmax_n) * 1.3
            with st.spinner("Computing VLP..."):
                df_vlp, vlp_kr, vlp_w = generate_vlp_curve(
                    Pwh=vlp_Pwh, TVD=vlp_TVD, d_tubing=vlp_d_tubing, api=api, gas_sg=gas_sg,
                    GOR=GOR if GOR > 0 else 500.0, Pb=Pb_computed, water_cut=water_cut, T_surface=vlp_T_surface,
                    T_bottom=T, water_sg=water_sg, roughness=vlp_roughness, q_max=vlp_qmax,
                    N=100, n_segments=int(vlp_segments), pvt_method=pvt_method, Psep=Psep, Tsep=Tsep, salinity=salinity,
                    use_polynomial=use_polynomial, friction_method=friction_method,
                    vlp_correlation=vlp_correlation,
                    MD=vlp_MD,
                    inclination_deg=bb_inclination_deg,
                    flow_direction=bb_flow_direction,
                    apply_payne=bb_apply_payne, 
                    apply_bo_undersat_correction=apply_bo_undersat_correction,
                    Rs_manual=Rs_manual if Rs_manual > 0 else None,
                    Bo_manual=Bo_manual if Bo_manual > 0 else None,
                    mu_o_manual=mu_o_manual if mu_o_manual > 0 else None,                   
                )
                
                # ── Fix Plot Autoscaling for Choked Flow ──────────────────────
                # If the well hits choked flow (velocity limit), friction shoots to infinity.
                # This is physically correct, but it ruins the Y-axis scale of the plot.
                # We cap the VLP plotted pressures to roughly 1.5x the max reservoir pressure.
                #==================================================================================
                # Keep raw VLP for operating-point calculation
                #==================================================================================
                df_vlp_plot = df_vlp

                if df_vlp is not None and df_ipr_n is not None:
                    max_p = df_ipr_n['Pwf (psia)'].max() * 1.5
                    df_vlp_plot = df_vlp.copy()
                    df_vlp_plot.loc[
                        df_vlp_plot['Pwf (psia)'] > max_p,
                        'Pwf (psia)'
                    ] = float('nan')

                    # Safety: if pressure cap hides the whole VLP curve, show raw VLP instead
                    if df_vlp_plot['Pwf (psia)'].notna().sum() < 2:
                        df_vlp_plot = df_vlp.copy()

    with tab_vlp_curve:
        if not enable_vlp:
            st.info("Enable **VLP / Nodal Analysis** in the sidebar.")

        elif vlp_sensitivity_mode and vlp_sensitivity_values and not errors:
            # ── VLP Sensitivity Mode ──────────────────────────────────
            st.markdown(f"#### VLP Sensitivity: {vlp_sensitivity_param}")
            vlp_sens_curves = []
            with st.spinner("Computing VLP sensitivity curves..."):
                for val in vlp_sensitivity_values:
                    sens_GOR = GOR if GOR > 0 else 500.0
                    sens_d_tubing = vlp_d_tubing

                    if "GOR" in vlp_sensitivity_param:
                        sens_GOR = val
                        label = f"GOR = {val:.0f} scf/STB"
                    else:
                        sens_d_tubing = val
                        label = f"ID = {val:.3f} in"

                    df_vlp_s, _, _ = generate_vlp_curve(
                        Pwh=vlp_Pwh, TVD=vlp_TVD, d_tubing=sens_d_tubing, api=api, gas_sg=gas_sg,
                        GOR=sens_GOR, Pb=Pb_computed, water_cut=water_cut, T_surface=vlp_T_surface,
                        T_bottom=T, water_sg=water_sg, roughness=vlp_roughness, q_max=vlp_qmax if 'vlp_qmax' in dir() else 3000.0,
                        N=100, n_segments=max(int(vlp_segments), 2), pvt_method=pvt_method, Psep=Psep, Tsep=Tsep, salinity=salinity,
                        use_polynomial=use_polynomial, friction_method=friction_method,
                        vlp_correlation=vlp_correlation,
                        MD=vlp_MD,
                        inclination_deg=bb_inclination_deg,
                        flow_direction=bb_flow_direction,
                        apply_payne=bb_apply_payne,
                        apply_bo_undersat_correction=apply_bo_undersat_correction,
                        Rs_manual=Rs_manual if Rs_manual > 0 else None,
                        Bo_manual=Bo_manual if Bo_manual > 0 else None,
                        mu_o_manual=mu_o_manual if mu_o_manual > 0 else None,
                    )
                    if df_vlp_s is not None:
                        vlp_sens_curves.append((label, df_vlp_s))

            if vlp_sens_curves:
                vlp_plot_label = vlp_kr.get("method", vlp_label) if vlp_kr else vlp_label
                fig_vlp_sens = plot_vlp_sensitivity(vlp_sens_curves, vlp_sensitivity_param, vlp_plot_label)
                st.plotly_chart(fig_vlp_sens, use_container_width=True)
            else:
                st.warning("Could not generate VLP sensitivity curves.")

        elif df_vlp_plot is not None:
            vlp_plot_label = vlp_kr.get("method", vlp_label) if vlp_kr else vlp_label
            fig_vlp_only = go.Figure()

            fig_vlp_only.add_trace(go.Scatter(
                x=df_vlp_plot['q (STB/day)'],
                y=df_vlp_plot['Pwf (psia)'],
                mode='lines',
                name=vlp_plot_label,
                line=dict(color='red')
            ))

            fig_vlp_only.update_layout(
                title=f"VLP Curve ({vlp_plot_label})",
                xaxis_title="Rate (STB/day)",
                yaxis_title="Pressure (psia)"
            )

            st.plotly_chart(fig_vlp_only, use_container_width=True)

        else:
            st.error("Could not generate VLP curve.")

    with tab_nodal:
        if not enable_vlp:
            st.info("Enable **VLP / Nodal Analysis** in the sidebar.")
        elif df_ipr_n is not None and df_vlp is not None:
            # ── Find operating point ──────────────────────────────
            op = find_operating_point(df_ipr_n, df_vlp)

            # ── Plot (uses the new plots.py function) ─────────────
            ipr_label = kr_n.get("method", ipr_method)
            vlp_plot_label = vlp_kr.get("method", vlp_label) if vlp_kr else vlp_label
            test_pt = None
            if qtest > 0 and Pwf_test > 0:
                test_pt = (qtest, Pwf_test)

            fig_nodal = plot_nodal_analysis(
                df_ipr_n, df_vlp_plot,
                operating_point=op,
                ipr_label=ipr_label,
                vlp_label=vlp_plot_label,
                test_point=test_pt,
            )
            st.plotly_chart(fig_nodal, use_container_width=True)

            # ── Results ───────────────────────────────────────────
            if op["found"]:
                st.markdown("#### Operating Point")
                c1, c2, c3 = st.columns(3)
                c1.metric("Flow Rate (q)", f"{op['q_op']:.1f} STB/day")
                c2.metric("Pwf at Node", f"{op['Pwf_op']:.1f} psia")
                c3.metric("Wellhead Pressure", f"{vlp_Pwh:.1f} psia")

                st.markdown("#### Summary")
                params_n, vals_n = [], []

                params_n.append("Operating Rate"); vals_n.append(f"{op['q_op']:.2f} STB/day")
                params_n.append("Operating Pwf"); vals_n.append(f"{op['Pwf_op']:.2f} psia")
                params_n.append("Drawdown"); vals_n.append(f"{Pr - op['Pwf_op']:.2f} psi")

                qmax_aof = kr_n.get("qmax_AOF", "N/A")
                if isinstance(qmax_aof, (int, float)) and qmax_aof > 0:
                    params_n.append("% of AOF"); vals_n.append(f"{op['q_op']/float(qmax_aof)*100:.1f}%")
                params_n.append("IPR qmax (AOF)"); vals_n.append(f"{qmax_aof} STB/day")
                params_n.append("Simplified Static BHP (q=0)")
                vals_n.append(f"{vlp_kr.get('Pwf_static','N/A')} psia")
                params_n.append("Wellhead Pressure"); vals_n.append(f"{vlp_Pwh:.2f} psia")
                params_n.append("TVD"); vals_n.append(f"{vlp_TVD:.1f} ft")
                if vlp_correlation == "Beggs-Brill":
                    params_n.append("MD"); vals_n.append(f"{vlp_MD:.1f} ft")
                    params_n.append("Inclination"); vals_n.append(f"{bb_inclination_deg:.2f}° from horizontal")
                    params_n.append("Flow Direction"); vals_n.append(bb_flow_direction)                
                params_n.append("Tubing ID"); vals_n.append(f"{vlp_d_tubing:.3f} in")
                params_n.append("IPR Method"); vals_n.append(ipr_label)
                params_n.append("VLP Correlation"); vals_n.append(vlp_kr.get("method", vlp_label))

                st.dataframe(
                    pd.DataFrame({"Parameter": params_n, "Value": vals_n}),
                    hide_index=True, use_container_width=True,
                )
                if vlp_kr and vlp_kr.get("Pwf_static_note"):
                    st.caption(f"{vlp_kr.get('Pwf_static_label', 'Simplified Static BHP')}: "
                               f"{vlp_kr.get('Pwf_static_note')}"
                    )
            else:
                st.warning(f"⚠ {op['message']}")
        else:
            st.error("Could not generate IPR/VLP curve for nodal analysis.")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 5: IPR CALCULATION
    # ══════════════════════════════════════════════════════════════════════
    with tab_ipr_table:
        if not errors:
            df_ipr_tab, kr_tab, _ = generate_ipr_curve(ipr_key, ipr_params)
            if df_ipr_tab is not None:
                st.markdown(f"#### IPR Data Table — {kr_tab.get('method', ipr_method)}")
                st.dataframe(df_ipr_tab, use_container_width=True, hide_index=True, height=500)

                csv_tab = df_ipr_tab.to_csv(index=False)
                st.download_button("Download Table (CSV)", csv_tab, "ipr_table.csv",
                                   "text/csv", use_container_width=True)
            else:
                st.info("Generate an IPR curve first.")
        else:
            st.error("Fix input errors first.")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 6: VLP CALCULATION
    # ══════════════════════════════════════════════════════════════════════
    with tab_vlp_table:
        if not enable_vlp:
            st.info("Enable **VLP / Nodal Analysis** in the sidebar.")
        elif errors:
            st.error("Fix input errors first.")
        else:
            with st.spinner("Calculating step-by-step pressure traverse..."):
                from backend.vlp import detailed_pressure_traverse
                
                # Use operating point if available, else test rate, else 1000
                operating_q = 1000.0
                if op is not None and op.get("found"):
                    operating_q = op["q_op"]
                elif qtest > 0:
                    operating_q = qtest

                df_detailed = detailed_pressure_traverse(
                    Pwh=vlp_Pwh,
                    q_oil=operating_q,
                    TVD=vlp_TVD,
                    d_tubing=vlp_d_tubing,
                    api=api,
                    gas_sg=gas_sg,
                    GOR=GOR if GOR > 0 else 500.0,
                    Pb=Pb_computed,
                    water_cut=water_cut,
                    T_surface=vlp_T_surface,
                    T_bottom=T,
                    roughness=vlp_roughness,
                    n_segments=int(vlp_segments),
                    pvt_method=pvt_method,
                    Psep=Psep,
                    Tsep=Tsep,
                    salinity=salinity,
                    use_polynomial=use_polynomial,
                    friction_method=friction_method,
                    vlp_correlation=vlp_correlation,
                    MD=vlp_MD,
                    inclination_deg=bb_inclination_deg,
                    flow_direction=bb_flow_direction,
                    apply_payne=bb_apply_payne, 
                    water_sg=water_sg,
                    apply_bo_undersat_correction=apply_bo_undersat_correction,     
                    Rs_manual=Rs_manual if Rs_manual > 0 else None,
                    Bo_manual=Bo_manual if Bo_manual > 0 else None,
                    mu_o_manual=mu_o_manual if mu_o_manual > 0 else None,              
                )
                
                st.markdown("#### Detailed VLP Stepwise Calculation")
                st.dataframe(df_detailed, use_container_width=True, hide_index=True, height=600)
                
                csv_vlp = df_detailed.to_csv(index=False)
                st.download_button("Download VLP Table (CSV)", csv_vlp, "vlp_detailed.csv", "text/csv", use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 7: ASSUMPTIONS & INFO
    # ══════════════════════════════════════════════════════════════════════
    with tab_assumptions:
        st.markdown("#### IPR Model Descriptions & Assumptions")

        with st.expander("Simple Darcy / Productivity Index IPR", expanded=False):
            st.markdown("""
            **Equation:** `q = J × (Pr − Pwf)`

            **Assumptions:**
            - Single-phase oil flow (above bubble point or approximation)
            - Steady-state or pseudo-steady-state radial flow
            - Homogeneous, isotropic reservoir
            - Newtonian fluid, incompressible flow (constant μo, Bo)
            - No turbulence (Darcy flow)

            **When to use:** Undersaturated oil reservoirs where Pwf > Pb throughout,
            or as a first approximation.

            **J from reservoir properties:**
            `J = kh / [141.2 × μo × Bo × (ln(re/rw) − 0.75 + s)]`
            """)

        with st.expander("Vogel IPR", expanded=False):
            st.markdown("""
            **Equation:** `q/qmax = 1 − 0.2×(Pwf/Pr) − 0.8×(Pwf/Pr)²`

            **Assumptions:**
            - Solution-gas-drive reservoir (saturated oil)
            - Reservoir pressure at or below bubble point (Pr ≤ Pb)
            - Empirical relationship based on simulation studies
            - No free gas cap, no water influx

            **When to use:** Saturated oil reservoirs where two-phase flow
            (oil + gas) occurs throughout the reservoir.

            **qmax from test data:**
            `qmax = qtest / [1 − 0.2×(Pwf_test/Pr) − 0.8×(Pwf_test/Pr)²]`
            """)

        with st.expander("Composite IPR (Darcy + Vogel)", expanded=False):
            st.markdown("""
            **Above Pb:** `q = J × (Pr − Pwf)` (linear / Darcy)

            **Below Pb:** `q = qb + (J×Pb/1.8) × [1 − 0.2×(Pwf/Pb) − 0.8×(Pwf/Pb)²]`

            Where `qb = J × (Pr − Pb)` is the flow rate at bubble point.

            **Assumptions:**
            - Reservoir pressure above bubble point (Pr > Pb)
            - Above Pb: single-phase oil, linear IPR
            - Below Pb: two-phase flow, Vogel-type behavior
            - Curve is continuous at bubble point

            **When to use:** Undersaturated oil reservoirs where Pr > Pb
            and well may flow below bubble point.
            """)

        with st.expander("Standing's IPR with Flow Efficiency (New)", expanded=False):
            st.markdown("""
            **Saturated Reservoir (Pr ≤ Pb):**
            - `Pwf' = Pr − FE × (Pr − Pwf)`
            - `q = qmax_ideal × [1 − 0.2×(Pwf'/Pr) − 0.8×(Pwf'/Pr)²]`

            **Undersaturated Reservoir (Pr > Pb):**
            - Above Pb: `q = J_new × (Pr − Pwf)` (linear)
            - Below Pb: `Pwf' = Pb − FE × (Pb − Pwf)`
            - `q = J_new*(Pr−Pb) + (J_new*Pb/1.8) × Vogel(Pwf'/Pb)`

            **Flow Efficiency:**
            - FE < 1: Damaged well (positive skin)
            - FE = 1: Ideal well (reduces to standard Vogel)
            - FE > 1: Stimulated well (negative skin, acidized, fractured)
            - `FE = ln(0.472*re/rw) / [ln(0.472*re/rw) + s]`

            **FE > 1 Pressure Limit:**
            - For stimulated wells (FE > 1), the IPR curve terminates at
              `Pwf_min = P_ref × (1 − 1/FE)` because below this pressure
              Pwf' becomes negative (non-physical).
            - Corrected qmax: `qmax = qmax_ideal × (0.624 + 0.376 × FE)`

            **When to use:** Evaluating the effect of well damage or
            stimulation on IPR. Works for both saturated and undersaturated
            reservoirs.
            """)

        with st.expander("Fetkovich IPR (New)", expanded=False):
            st.markdown("""
            **Core Equation:** `q = C × (Pr² − Pwf²)^n`

            **Exponent Determination:**
            - From two test points: `n = ln(q₁/q₂) / ln(Δ₁/Δ₂)`
              where `Δ₁ = Pr² − Pwf₁²`, `Δ₂ = Pr² − Pwf₂²`
            - User-specified: any value in [0.5, 1.0]
            - Default: n = 1.0 (pure Darcy / laminar flow)

            **Derived Parameters:**
            - `C = q_test / (Pr² − Pwf_test²)^n`
            - `J = [C × (Pr² − Pwf_test²)^n] / (Pr − Pwf_test)`
            - `qmax = C × Pr^(2n)`

            **Parameters:**
            - n = 1.0: Darcy (laminar) flow
            - n = 0.5: Turbulent (non-Darcy) flow

            **When to use:** Oil or gas wells where back-pressure-type
            deliverability testing has been performed.  Multi-rate tests
            allow solving for n analytically.
            """)

        st.markdown("---")
        st.markdown("#### VLP Model Descriptions & Assumptions")

        with st.expander("Modified Hagedorn-Brown", expanded=False):
            st.markdown("""
            **Description:** An empirical, generalized two-phase flow correlation derived from a 1,500 ft experimental well.
            
            **Assumptions:**
            - Steady-state two-phase flow in vertical upward conduits.
            - Considers slip between phases (liquid holdup) but does not distinguish between distinct flow regimes initially (treats flow regime transitions implicitly).
            - Based on an original dataset involving water, oil, and gas mixtures.
            - Pressure gradient consists of hydrostatic (elevation), friction, and acceleration terms.
            - "Modified" refers to the integration of the Griffith correlation for bubble flow to improve accuracy at low flow rates.

            **When to use:** Widely accepted as a standard correlation for vertical oil wells, especially in slug and transition flow regimes.
            """)

        with st.expander("Beggs & Brill", expanded=False):
            st.markdown("""
            **Description:** An empirical correlation that accounts for flow at any inclination angle from horizontal to vertical.
            
            **Assumptions:**
            - Identifies the flow regime that would exist if the pipe were horizontal, then calculates liquid holdup.
            - Corrects the horizontal liquid holdup for pipe inclination using empirical factors.
            - Includes hydrostatic, friction, and acceleration pressure losses.
            - Flow regimes modeled: Segregated (stratified, wavy, annular), Intermittent (slug, plug), and Distributed (bubble, mist).

            **When to use:** Applicable to deviated and horizontal wells, pipelines, and when dealing with multiphase flow in varying topography. Includes the Payne correction for improved accuracy in upward vertical flow.
            """)

        st.markdown("---")
        st.markdown("#### PVT Correlation Notes")

        with st.expander("PVT Correlations Summary", expanded=False):
            st.markdown("""
            | Property | Correlation | Validity Range |
            |----------|-------------|----------------|
            | Rs | Standing (1947) | API 16–58, T 100–258°F |
            | Rs | Vasquez-Beggs (1980) | Wide range |
            | Pb | Standing / Vasquez-Beggs | Same as Rs |
            | Bo | Standing (1947) | Same as Rs |
            | Bo | Vasquez-Beggs (1980) | Wide range |
            | μo (dead) | Beggs-Robinson (1975) | API 16–58, T 70–295°F |
            | μo (live) | Beggs-Robinson (1975) | — |
            | Z-factor | Dranchuk-Abou-Kassem (1975) | Tpr 1–3, Ppr 0.2–30 |
            | μg | Lee-Gonzalez-Eakin (1966) | Natural gas |
            | Bw | McCain | — |

            **Important:** All correlations are empirical and have validity limits.
            Results are engineering estimates, not replacements for calibrated models.
            """)

        st.markdown("---")
        st.markdown("#### General Disclaimer")
        st.markdown("""
        <div class="info-box">
            <strong>Engineering Estimates Only</strong><br>
            Flow Assure uses published empirical correlations to estimate PVT properties
            and generate IPR curves. These results are approximations suitable for
            preliminary engineering analysis and educational purposes.
            They should not replace laboratory PVT reports, well-test analysis,
            or calibrated reservoir simulation models for production decisions.
        </div>
        """, unsafe_allow_html=True)
