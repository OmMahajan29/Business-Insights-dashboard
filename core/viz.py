from typing import Dict, Any, Tuple
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from .constants import ALLOWED_CHARTS
from .plan import AnalysisPlan

def coerce_list(x):
    return x or []

def choose_chart(plan: AnalysisPlan, result: pd.DataFrame, override: str) -> Tuple[str, Dict[str, Any]]:
    chart = plan.chart_suggestion if plan.chart_suggestion in ALLOWED_CHARTS else None
    if override and override != "Auto":
        chart = override

    dims = coerce_list(plan.dimensions)
    mets = coerce_list(plan.metrics)
    fig_kwargs: Dict[str, Any] = {}

    if not chart:
        if len(dims) == 1 and len(mets) == 1:
            chart = "bar"
        elif len(dims) == 1 and len(mets) >= 2:
            chart = "line"
        elif len(dims) == 0 and len(mets) == 1:
            chart = "hist"
        else:
            chart = "bar"

    # Build fig kwargs
    if chart == "hist":
        # Histogram: x is the metric column; the executor produced summary stats OR we can use raw column name
        if len(mets) == 1:
            x_col = mets[0]
            if x_col not in result.columns:
                # If summary stats table, attempt to locate the metric column in original plan
                # Fall back to numeric-looking column
                numeric_cols = result.select_dtypes(include="number").columns.tolist()
                x_col = x_col if x_col in result.columns else (numeric_cols[0] if numeric_cols else result.columns[-1])
            fig_kwargs = {"x": x_col}
        else:
            # fallback
            fig_kwargs = {"x": result.columns[-1]}
    else:
        if len(dims) >= 1 and len(mets) >= 1:
            x_col = dims[0] if dims[0] in result.columns else result.columns[0]
            y_col = mets[0]
            if y_col not in result.columns:
                numeric_cols = result.select_dtypes(include="number").columns.tolist()
                y_col = numeric_cols[0] if numeric_cols else result.columns[-1]
            fig_kwargs = {"x": x_col, "y": y_col}
        elif len(mets) == 1:
            y_col = mets[0]
            if y_col not in result.columns:
                numeric_cols = result.select_dtypes(include="number").columns.tolist()
                y_col = numeric_cols[0] if numeric_cols else result.columns[-1]
            fig_kwargs = {"x": y_col}
        else:
            fig_kwargs = {"x": result.columns[0], "y": result.columns[-1]}

    return chart, fig_kwargs

def render_chart(chart: str, data: pd.DataFrame, fig_kwargs: Dict[str, Any], color_by: str = ""):
    if color_by and color_by in data.columns and chart in {"bar", "line", "area", "scatter"}:
        fig_kwargs = {**fig_kwargs, "color": color_by}

    if chart == "bar":
        fig = px.bar(data, **fig_kwargs)
    elif chart == "line":
        fig = px.line(data, **fig_kwargs)
    elif chart == "area":
        fig = px.area(data, **fig_kwargs)
    elif chart == "scatter":
        fig = px.scatter(data, **fig_kwargs)
    elif chart == "hist":
        fig = px.histogram(data, **fig_kwargs)
    elif chart == "heatmap":
        fig = px.imshow(data.select_dtypes(include="number").corr())
    else:
        fig = px.bar(data, **fig_kwargs)

    st.plotly_chart(fig, use_container_width=True)
    return fig

def make_downloads(result: pd.DataFrame, fig, download_name: str):
    st.subheader("Exports")
    csv_bytes = result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download data (CSV)",
        data=csv_bytes,
        file_name=f"{download_name}.csv",
        mime="text/csv",
    )
    try:
        img_bytes = pio.to_image(fig, format="png", scale=2)
        st.download_button(
            "Download chart (PNG)",
            data=img_bytes,
            file_name=f"{download_name}.png",
            mime="image/png",
        )
    except Exception:
        st.caption("To enable PNG export, install kaleido and ensure Chrome is available.")
