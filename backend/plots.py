from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────────────────────────────────────────────────────────
# Color Palette
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "Darcy (PI)":       "#3B82F6",   # blue
    "Vogel":            "#10B981",   # emerald
    "Composite":        "#F59E0B",   # amber
    "Standing":         "#8B5CF6",   # violet
    "Fetkovich":        "#EF4444",   # red
    "primary":          "#3B82F6",
    "secondary":        "#10B981",
    "accent":           "#F59E0B",
    "operating_point":  "#EF4444",
    "bubble_point":     "#8B5CF6",
}

METHOD_COLOR_MAP = {
    "Darcy (PI)": "#3B82F6",
    "Vogel": "#10B981",
    "Composite": "#F59E0B",
    "Standing": "#8B5CF6",
    "Fetkovich": "#EF4444",
}


def _get_color_for_method(method_label: str) -> str:
    """Get color for a given method label string."""
    for key, color in METHOD_COLOR_MAP.items():
        if key.lower() in method_label.lower():
            return color
    return "#6B7280"  # gray fallback


# ──────────────────────────────────────────────────────────────────────────────
# Single IPR Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_ipr_curve(
    df: pd.DataFrame,
    method_name: str,
    operating_point: Optional[tuple[float, float]] = None,
    bubble_point: Optional[tuple[float, float]] = None,
    key_results: Optional[dict] = None,
) -> go.Figure:
    """
    Create an interactive IPR curve plot.

    Parameters
    ----------
    df              : DataFrame with 'Pwf (psia)' and 'q (STB/day)' columns
    method_name     : Name of the IPR method for the title
    operating_point : (q, Pwf) tuple for marking the test/operating point
    bubble_point    : (q, Pb) tuple for marking the bubble point (Composite IPR)
    key_results     : dict of key results to annotate

    Returns
    -------
    go.Figure
    """
    color = _get_color_for_method(method_name)

    fig = go.Figure()

    # Main curve
    fig.add_trace(go.Scatter(
        x=df["q (STB/day)"],
        y=df["Pwf (psia)"],
        mode="lines",
        name=method_name,
        line=dict(color=color, width=3),
        hovertemplate=(
            "<b>q</b> = %{x:.1f} STB/day<br>"
            "<b>Pwf</b> = %{y:.1f} psia<br>"
            "<extra></extra>"
        ),
    ))

    # Operating point
    if operating_point is not None:
        q_op, Pwf_op = operating_point
        fig.add_trace(go.Scatter(
            x=[q_op],
            y=[Pwf_op],
            mode="markers+text",
            name="Operating Point",
            marker=dict(size=14, color=COLORS["operating_point"], symbol="circle",
                        line=dict(width=2, color="white")),
            text=[f"  ({q_op:.0f}, {Pwf_op:.0f})"],
            textposition="top right",
            textfont=dict(size=12, color=COLORS["operating_point"]),
            hovertemplate=(
                "<b>Test Point</b><br>"
                f"q = {q_op:.1f} STB/day<br>"
                f"Pwf = {Pwf_op:.1f} psia<br>"
                "<extra></extra>"
            ),
        ))

    # Bubble point
    if bubble_point is not None:
        q_bp, Pwf_bp = bubble_point
        fig.add_trace(go.Scatter(
            x=[q_bp],
            y=[Pwf_bp],
            mode="markers+text",
            name="Bubble Point",
            marker=dict(size=14, color=COLORS["bubble_point"], symbol="diamond",
                        line=dict(width=2, color="white")),
            text=[f"  Pb ({q_bp:.0f}, {Pwf_bp:.0f})"],
            textposition="top right",
            textfont=dict(size=12, color=COLORS["bubble_point"]),
            hovertemplate=(
                "<b>Bubble Point</b><br>"
                f"qb = {q_bp:.1f} STB/day<br>"
                f"Pb = {Pwf_bp:.1f} psia<br>"
                "<extra></extra>"
            ),
        ))

    # Layout
    fig.update_layout(
        title=dict(
            text=f"IPR Curve — {method_name}",
            font=dict(size=20, family="Inter, sans-serif"),
            x=0.5,
            xanchor="center",
            pad=dict(b=30),  # <--- 1. ADDED PADDING HERE TO PUSH LEGEND DOWN
        ),
        xaxis=dict(
            title="Oil Flow Rate, q (STB/day)",
            title_font=dict(size=14),
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            rangemode="tozero",
            rangeslider=dict(visible=True),  # <--- 2. ADDED RANGE SLIDER HERE
        ),
        yaxis=dict(
            title="Flowing Bottom-Hole Pressure, Pwf (psia)",
            title_font=dict(size=14),
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True,
            rangemode="tozero",
        ),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        height=550,
        hovermode="closest",
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Comparison Plot (Multiple IPR Methods)
# ──────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    curves: dict[str, pd.DataFrame],
    operating_point: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """
    Overlay multiple IPR curves on one chart for comparison.

    Parameters
    ----------
    curves          : dict mapping method name → DataFrame
    operating_point : optional (q, Pwf) to mark
    """
    fig = go.Figure()

    for name, df in curves.items():
        color = _get_color_for_method(name)
        fig.add_trace(go.Scatter(
            x=df["q (STB/day)"],
            y=df["Pwf (psia)"],
            mode="lines",
            name=name,
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "q = %{x:.1f} STB/day<br>"
                "Pwf = %{y:.1f} psia<br>"
                "<extra></extra>"
            ),
        ))

    if operating_point is not None:
        q_op, Pwf_op = operating_point
        fig.add_trace(go.Scatter(
            x=[q_op], y=[Pwf_op],
            mode="markers",
            name="Test Point",
            marker=dict(size=14, color=COLORS["operating_point"], symbol="circle",
                        line=dict(width=2, color="white")),
        ))

    fig.update_layout(
        title=dict(
            text="IPR Curve Comparison",
            font=dict(size=20, family="Inter, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Oil Flow Rate, q (STB/day)",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        yaxis=dict(
            title="Flowing Bottom-Hole Pressure, Pwf (psia)",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        template="plotly_white",
        
        # --- NEW LEGEND SETTINGS: VERTICAL ON THE RIGHT SIDE ---
        legend=dict(
            orientation="v",       # 'v' stands for vertical (line by line)
            yanchor="top",         # Anchor it to the top edge
            y=1.0,                 # Align with the top of the plotting area
            xanchor="left",        # Anchor it to its own left edge
            x=1.02                 # Push it just past the right edge of the graph (1.0)
        ),
        
        # --- INCREASED RIGHT MARGIN (r) TO 250 SO LEGEND TEXT FITS ---
        margin=dict(l=60, r=250, t=80, b=60),
        
        height=550,
        hovermode="closest",
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity Analysis Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(
    curves: list[tuple[str, pd.DataFrame]],
    param_name: str,
    method_name: str,
) -> go.Figure:
    """
    Plot a family of IPR curves showing sensitivity to a single parameter.

    Parameters
    ----------
    curves      : list of (label, DataFrame) tuples
    param_name  : Name of the parameter being varied (for title)
    method_name : IPR method name
    """
    fig = go.Figure()

    n = len(curves)
    # Generate a colorscale from blue to red
    for i, (label, df) in enumerate(curves):
        t = i / max(n - 1, 1)
        r = int(59 + t * (239 - 59))
        g = int(130 - t * (130 - 68))
        b = int(246 - t * (246 - 68))
        color = f"rgb({r},{g},{b})"

        fig.add_trace(go.Scatter(
            x=df["q (STB/day)"],
            y=df["Pwf (psia)"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=dict(
            text=f"Sensitivity Analysis — {param_name} ({method_name})",
            font=dict(size=18, family="Inter, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Oil Flow Rate, q (STB/day)",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        yaxis=dict(
            title="Pwf (psia)",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        margin=dict(l=60, r=150, t=80, b=60),
        height=550,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# VLP Sensitivity Analysis Plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_vlp_sensitivity(
    curves: list[tuple[str, pd.DataFrame]],
    param_name: str,
    vlp_method: str,
) -> go.Figure:
    """
    Plot a family of VLP curves showing sensitivity to a single parameter.

    Parameters
    ----------
    curves      : list of (label, DataFrame) tuples
    param_name  : Name of the parameter being varied (for title)
    vlp_method  : VLP correlation name
    """
    fig = go.Figure()

    n = len(curves)
    # Generate a colorscale from blue to red
    for i, (label, df) in enumerate(curves):
        t = i / max(n - 1, 1)
        r = int(59 + t * (239 - 59))
        g = int(130 - t * (130 - 68))
        b = int(246 - t * (246 - 68))
        color = f"rgb({r},{g},{b})"

        fig.add_trace(go.Scatter(
            x=df["q (STB/day)"],
            y=df["Pwf (psia)"],
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        title=dict(
            text=f"VLP Sensitivity — {param_name} ({vlp_method})",
            font=dict(size=18, family="Inter, sans-serif"),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            title="Oil Flow Rate, q (STB/day)",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        yaxis=dict(
            title="Flowing Bottom-Hole Pressure, Pwf (psia)",
            gridcolor="rgba(128,128,128,0.2)",
            rangemode="tozero",
        ),
        template="plotly_white",
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02),
        margin=dict(l=60, r=150, t=80, b=60),
        height=550,
    )

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Nodal Analysis Plot (IPR + VLP)
# ──────────────────────────────────────────────────────────────────────────────

def plot_nodal_analysis(
    df_ipr: pd.DataFrame,
    df_vlp: pd.DataFrame,
    operating_point: Optional[dict] = None,
    ipr_label: str = "IPR",
    vlp_label: str = "VLP",
    test_point: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """Overlay IPR and VLP curves with the operating-point intersection."""
    fig = go.Figure()

    # IPR Trace
    fig.add_trace(go.Scatter(
        x=df_ipr["q (STB/day)"], y=df_ipr["Pwf (psia)"],
        mode="lines", name=f"IPR — {ipr_label}",
        line=dict(color="#3B82F6", width=3),
        hovertemplate="<b>IPR</b><br>q=%{x:.1f}<br>Pwf=%{y:.1f}<extra></extra>",
    ))

    # VLP Trace
    fig.add_trace(go.Scatter(
        x=df_vlp["q (STB/day)"], y=df_vlp["Pwf (psia)"],
        mode="lines", name=f"VLP — {vlp_label}",
        line=dict(color="#10B981", width=3),
        hovertemplate="<b>VLP</b><br>q=%{x:.1f}<br>Pwf=%{y:.1f}<extra></extra>",
    ))

    # Operating point Marker
    if operating_point and operating_point.get("found"):
        q_op = operating_point["q_op"]
        P_op = operating_point["Pwf_op"]
        fig.add_trace(go.Scatter(
            x=[q_op], y=[P_op],
            mode="markers+text",
            name="Operating Point",
            marker=dict(size=16, color="#F59E0B", symbol="star",
                        line=dict(width=2, color="white")),
            text=[f"  OP ({q_op:.0f}, {P_op:.0f})"],
            textposition="top right",
            textfont=dict(size=13, color="#F59E0B"),
            hovertemplate=(
                f"<b>Operating Point</b><br>"
                f"q = {q_op:.1f} STB/day<br>"
                f"Pwf = {P_op:.1f} psia<extra></extra>"
            ),
        ))

    # Test point Marker (Optional)
    if test_point is not None:
        q_t, P_t = test_point
        fig.add_trace(go.Scatter(
            x=[q_t], y=[P_t],
            mode="markers+text",
            name="Test Point",
            marker=dict(size=12, color="#EF4444", symbol="circle",
                        line=dict(width=2, color="white")),
            text=[f"  Test ({q_t:.0f}, {P_t:.0f})"],
            textposition="bottom right",
            textfont=dict(size=11, color="#EF4444"),
        ))

    # Clean layout exactly matching your UI
    fig.update_layout(
        title=dict(
            text="Nodal Analysis — IPR vs VLP",
            font=dict(size=20, family="Inter, sans-serif"),
            x=0.5, xanchor="center", pad=dict(b=30),
        ),
        xaxis=dict(
            title="Oil Flow Rate, q (STB/day)",
            title_font=dict(size=14),
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True, rangemode="tozero",
        ),
        yaxis=dict(
            title="Flowing Bottom-Hole Pressure, Pwf (psia)",
            title_font=dict(size=14),
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=True, rangemode="tozero",
        ),
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
        ),
        margin=dict(l=60, r=30, t=80, b=60),
        height=600,
        hovermode="closest",
    )

    return fig
