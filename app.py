import os
from typing import List, Optional, Literal, Dict, Any, Tuple

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# =========================
# Bootstrapping & Settings
# =========================
load_dotenv()
OR_KEY = os.getenv("OPENROUTER_API_KEY")

st.set_page_config(page_title="Business Insights Dashboard", layout="wide")

# Hard limits for safety
ROW_LIMIT_HARD = 5000
DISPLAY_PREVIEW_ROWS = 25
SAMPLE_THRESHOLD = 500_000
SAMPLE_SIZE = 200_000
ALLOWED_OPS = {"=", "!=", ">", "<", ">=", "<=", "in", "not in", "contains", "startswith", "endswith"}
ALLOWED_AGGS = {"sum", "avg", "count", "min", "max"}
ALLOWED_CHARTS = {"bar", "line", "area", "scatter", "hist", "heatmap"}

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "last_plan" not in st.session_state:
    st.session_state.last_plan = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_chart" not in st.session_state:
    st.session_state.last_chart = None
if "exec_trace" not in st.session_state:
    st.session_state.exec_trace = []

st.title("LLM-Powered Business Insights Dashboard")

# =========================
# Sidebar: Data Source + Controls
# =========================
st.sidebar.header("Data Source")
mode = st.sidebar.radio(
    "Choose data mode",
    options=["CSV (upload)", "SQL (coming soon)"],
    index=0
)

@st.cache_data(show_spinner=False)
def read_csv_preview(_file, _sep, _header, _enc, _nrows):
    return pd.read_csv(_file, sep=_sep, header=_header, encoding=_enc, nrows=int(_nrows))

def maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > SAMPLE_THRESHOLD:
        return df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    return df

if mode == "CSV (upload)":
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    read_opts = st.sidebar.expander("CSV read options", expanded=False)
    with read_opts:
        sep = st.text_input("Delimiter (sep)", value=",")
        header_row = st.number_input("Header row index", min_value=0, value=0, step=1)
        encoding = st.text_input("Encoding", value="utf-8")
        nrows_preview = st.number_input("Preview rows", min_value=10, max_value=5000, value=200, step=10)

    if file is not None:
        try:
            _ = read_csv_preview(file, sep, header_row, encoding, nrows_preview)
            file.seek(0)
            df_full = pd.read_csv(file, sep=sep, header=header_row, encoding=encoding)
            df_full = maybe_sample(df_full)
            st.session_state.df = df_full
            st.session_state.dataset_name = getattr(file, "name", "uploaded.csv")
            st.sidebar.success(
                f"Loaded: {st.session_state.dataset_name} "
                f"(rows: {len(df_full):,}, cols: {df_full.shape[1]})"
            )
        except Exception as e:
            st.session_state.df = None
            st.sidebar.error(f"Failed to read CSV: {e}")
else:
    st.sidebar.info("SQL connector coming soon (read-only, SELECT-only).")

# Sidebar: Chart overrides
st.sidebar.header("Chart Options")
chart_override = st.sidebar.selectbox(
    "Chart type (optional override)",
    options=["Auto", "bar", "line", "area", "scatter", "hist", "heatmap"],
    index=0
)
color_by = st.sidebar.text_input("Color by (optional dimension)", value="")
download_name = st.sidebar.text_input("Download base name", value="insights_result")

# Status banners
if OR_KEY:
    st.success("LLM status: OpenRouter key loaded.")
else:
    st.warning("LLM status: No API key found. Interpretation uses the key; execution is local.")

if st.session_state.df is None:
    st.info("Upload a CSV in the sidebar to begin.")
else:
    st.success(f"Dataset ready: {st.session_state.dataset_name}")

# =========================
# Dataset Overview
# =========================
if st.session_state.df is not None:
    df = st.session_state.df

    st.subheader("Preview")
    st.dataframe(df.head(DISPLAY_PREVIEW_ROWS), use_container_width=True)

    st.subheader("Schema summary")
    dtypes = df.dtypes.astype(str)
    non_null_counts = df.notna().sum()
    null_counts = df.isna().sum()
    sample_vals = df.head(3).astype(str)

    schema_rows = []
    for col in df.columns:
        schema_rows.append({
            "column": col,
            "dtype": dtypes[col],
            "non_null": int(non_null_counts[col]),
            "nulls": int(null_counts[col]),
            "unique": int(df[col].nunique(dropna=True)),
            "example_values": ", ".join(sample_vals[col].tolist())
        })
    schema_df = pd.DataFrame(schema_rows)
    st.dataframe(schema_df, use_container_width=True)

# ======================================================
# Plan Model
# ======================================================
AggType = Literal["sum", "avg", "count", "min", "max"]
SortDir = Literal["asc", "desc"]

class FilterSpec(BaseModel):
    column: str
    op: Literal["=", "!=", ">", "<", ">=", "<=", "in", "not in", "contains", "startswith", "endswith"]
    value: Any

class AnalysisPlan(BaseModel):
    intent: str = Field(..., description="Short description of what the user wants.")
    metrics: List[str] = Field(default_factory=list)
    metric_aggs: Dict[str, AggType] = Field(default_factory=dict, description="Aggregation per metric.")
    dimensions: List[str] = Field(default_factory=list)
    timeframe: Optional[str] = Field(default=None)
    # IMPORTANT: Keep as Dict[str, SortDir] but we will normalize inputs before creating this model
    sort: Optional[Dict[str, SortDir]] = Field(default=None, description='e.g., {"Sales":"desc"}')
    filters: List[FilterSpec] = Field(default_factory=list)
    limit: Optional[int] = Field(default=20)
    chart_suggestion: Optional[str] = Field(default=None, description="bar|line|area|scatter|hist|heatmap")
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    notes: Optional[str] = None

def dataset_schema_summary(df: pd.DataFrame) -> str:
    cols = [f"{c} ({t})" for c, t in zip(df.columns, df.dtypes.astype(str).tolist())]
    return ", ".join(cols)

# -------- Interpreter (same as Day 3), but with stronger instruction on sort ----------
def call_interpreter_llm(user_question: str, df: pd.DataFrame) -> str:
    import requests
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    model = "mistralai/mistral-7b-instruct:free"
    schema_text = dataset_schema_summary(df)
    system_prompt = (
        "You are a data analysis planner. "
        "Output STRICT JSON for AnalysisPlan with these exact fields. "
        "For the 'sort' field, ALWAYS use a single-key object mapping COLUMN to DIRECTION, e.g. {\"Sales\":\"desc\"}. "
        "Do NOT use {\"by\":\"...\",\"dir\":\"...\"}.\n"
        "{\n"
        '  "intent": str,\n'
        '  "metrics": [str],\n'
        '  "metric_aggs": {str: "sum"|"avg"|"count"|"min"|"max"},\n'
        '  "dimensions": [str],\n'
        '  "timeframe": str|null,\n'
        '  "filters": [{"column": str, "op": "="|"!="|">"|"<"|">="|"<="|"in"|"not in"|"contains"|"startswith"|"endswith", "value": any}],\n'
        '  "sort": {str: "asc"|"desc"}|null,\n'
        '  "limit": int|null,\n'
        '  "chart_suggestion": "bar"|"line"|"area"|"scatter"|"hist"|"heatmap"|null,\n'
        '  "needs_clarification": bool,\n'
        '  "clarification_question": str|null,\n'
        '  "notes": str|null\n'
        "}\n"
        "Rules: Only use provided columns; prefer small limits (<=50); output JSON only."
    )
    user_prompt = (
        f"Dataset schema: {schema_text}\n\n"
        f"User question: {user_question}\n\n"
        "Return only the JSON."
    )
    headers = {
        "Authorization": f"Bearer {OR_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Business Insights Dashboard"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 600
    }
    resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def normalize_sort(obj: Any) -> Optional[Dict[str, str]]:
    """
    Accepts variations and returns a normalized mapping {column: 'asc'|'desc'} or None.
    Allowed inputs:
    - {"Sales": "desc"}
    - {"by": "Sales", "dir": "desc"}
    - {"column": "Sales", "direction": "desc"}
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        # case 1: already correct
        if len(obj) == 1:
            k, v = list(obj.items())
            vv = str(v).lower()
            if vv in {"asc", "desc"}:
                return {k: vv}
        # case 2: by/dir or column/direction
        by = obj.get("by") or obj.get("column")
        dir_ = obj.get("dir") or obj.get("direction")
        if by and isinstance(dir_, str) and dir_.lower() in {"asc", "desc"}:
            return {by: dir_.lower()}
    # unsupported shape
    return None

def parse_plan(text: str) -> AnalysisPlan:
    import json
    s = text.strip()
    lines = s.splitlines()
    if lines:
        first = lines.strip().lower()
        if "json" in first and len(first) <= 10:
            lines = lines[1:]
    if lines:
        last = lines[-1].strip()
        if len(last) <= 5 and not any(ch.isalnum() for ch in last):
            trial = "\n".join(lines[:-1]).strip()
            try:
                _ = json.loads(trial)
                lines = lines[:-1]
            except Exception:
                pass
    s = "\n".join(lines).strip()
    plan_raw = json.loads(s)

    # Normalize sort BEFORE model construction
    if "sort" in plan_raw:
        norm = normalize_sort(plan_raw.get("sort"))
        plan_raw["sort"] = norm

    return AnalysisPlan(**plan_raw)

# ======================================================
# Executor (unchanged from Day 4, but tolerant to sort)
# ======================================================
class ExecutionError(Exception):
    pass

def validate_columns_exist(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ExecutionError(f"Unknown column(s): {missing}")

def enforce_numeric(df: pd.DataFrame, metrics: List[str]) -> None:
    bad = [m for m in metrics if m in df.columns and not pd.api.types.is_numeric_dtype(df[m])]
    if bad:
        raise ExecutionError(f"Metrics must be numeric for aggregation: {bad}")

def coerce_list(x: Optional[List[str]]) -> List[str]:
    return x or []

def apply_filters(df: pd.DataFrame, filters: List[FilterSpec], trace: List[str]) -> pd.DataFrame:
    out = df
    for f in filters:
        if f.op not in ALLOWED_OPS:
            raise ExecutionError(f"Disallowed filter op: {f.op}")
        if f.column not in out.columns:
            raise ExecutionError(f"Unknown column in filter: {f.column}")
        col = f.column
        val = f.value
        if f.op == "=":
            out = out[out[col] == val]
        elif f.op == "!=":
            out = out[out[col] != val]
        elif f.op == ">":
            out = out[out[col] > val]
        elif f.op == "<":
            out = out[out[col] < val]
        elif f.op == ">=":
            out = out[out[col] >= val]
        elif f.op == "<=":
            out = out[out[col] <= val]
        elif f.op == "in":
            if not isinstance(val, list):
                raise ExecutionError("Filter 'in' requires a list value.")
            out = out[out[col].isin(val)]
        elif f.op == "not in":
            if not isinstance(val, list):
                raise ExecutionError("Filter 'not in' requires a list value.")
            out = out[~out[col].isin(val)]
        elif f.op == "contains":
            out = out[out[col].astype(str).str.contains(str(val), na=False)]
        elif f.op == "startswith":
            out = out[out[col].astype(str).str.startswith(str(val), na=False)]
        elif f.op == "endswith":
            out = out[out[col].astype(str).str.endswith(str(val), na=False)]
        trace.append(f"Filter: {col} {f.op} {val}")
    return out

def aggregate_df(df: pd.DataFrame, plan: AnalysisPlan, trace: List[str]) -> pd.DataFrame:
    dims = coerce_list(plan.dimensions)
    mets = coerce_list(plan.metrics)
    aggs = plan.metric_aggs or {}

    validate_columns_exist(df, dims + mets)
    enforce_numeric(df, mets)

    agg_map: Dict[str, str] = {}
    for m in mets:
        agg = aggs.get(m, "sum")
        if agg not in ALLOWED_AGGS:
            raise ExecutionError(f"Disallowed aggregation: {agg}")
        agg_map[m] = "mean" if agg == "avg" else agg

    work = df.copy()
    if plan.filters:
        work = apply_filters(work, plan.filters, trace)

    if dims:
        grouped = work.groupby(dims, dropna=False)
        result = grouped.agg(agg_map).reset_index()
        trace.append(f"Group by: {dims}; Aggregate: {agg_map}")
    else:
        result = work.agg(agg_map)
        if isinstance(result, pd.Series):
            result = result.to_frame().T
        result.insert(0, "all_rows", "all")
        trace.append(f"Aggregate: {agg_map}")

    # Tolerate either sort shapes (already normalized in parse, but double-guard)
    if plan.sort:
        try:
            sort_col, sort_dir = list(plan.sort.items())[0]
        except Exception:
            raise ExecutionError("Invalid sort format; expected {column: 'asc'|'desc'}.")
        if sort_col not in result.columns:
            raise ExecutionError(f"Sort column not found in result: {sort_col}")
        result = result.sort_values(by=sort_col, ascending=(sort_dir == "asc"))
        trace.append(f"Sort: {sort_col} {sort_dir}")

    limit = plan.limit or 20
    limit = max(1, min(limit, ROW_LIMIT_HARD))
    result = result.head(limit)
    trace.append(f"Limit: {limit} (hard cap {ROW_LIMIT_HARD})")

    return result

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
    if len(dims) >= 1 and len(mets) >= 1:
        x_col = dims[0] if dims in result.columns else result.columns
        y_col = mets if mets in result.columns else (
            result.select_dtypes(include="number").columns.tolist()
            if len(result.select_dtypes(include="number").columns) else result.columns[-1]
        )
        fig_kwargs = {"x": x_col, "y": y_col}
    elif len(mets) == 1:
        y_col = mets if mets in result.columns else (
            result.select_dtypes(include="number").columns.tolist()
            if len(result.select_dtypes(include="number").columns) else result.columns[-1]
        )
        fig_kwargs = {"x": y_col}
    else:
        fig_kwargs = {"x": result.columns, "y": result.columns[-1]}
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

# =========================
# Chat UI (interpret + execute)
# =========================
st.subheader("Chat")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Ask about your data…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.df is None:
        reply = "Please upload a CSV first in the sidebar, then ask a question about it."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        if not OR_KEY:
            reply = (
                "Interpreter unavailable (no API key). "
                "Add OPENROUTER_API_KEY to your .env to enable interpretation."
            )
            with st.chat_message("assistant"):
                st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Interpreting your question into a structured plan..."):
                    try:
                        raw = call_interpreter_llm(prompt, st.session_state.df)
                        import json
                        # Parse and normalize
                        s = raw.strip().replace("\r", "")
                        if not (s.startswith("{") and s.endswith("}")):
                            # try to drop possible first/last fence-ish lines
                            lines = s.splitlines()
                            if lines and "json" in lines[0].lower():
                                lines = lines[1:]
                            if lines and len(lines[-1].strip()) <= 5 and not any(ch.isalnum() for ch in lines[-1]):
                                lines = lines[:-1]
                            s = "\n".join(lines).strip()
                        plan_raw = json.loads(s)
                        plan_raw["sort"] = normalize_sort(plan_raw.get("sort"))
                        plan = AnalysisPlan(**plan_raw)

                        st.session_state.last_plan = plan.dict()
                        st.success("Analysis plan created.")
                        st.write("Plan (validated):")
                        st.json(plan.dict())

                        st.session_state.exec_trace = []
                        with st.spinner("Executing plan with safety checks..."):
                            try:
                                result = aggregate_df(st.session_state.df, plan, st.session_state.exec_trace)
                                st.session_state.last_result = result

                                st.subheader("Result")
                                st.dataframe(result, use_container_width=True)

                                chart, fig_kwargs = choose_chart(plan, result, chart_override)
                                st.subheader(f"Chart ({chart})")
                                fig = render_chart(chart, result, fig_kwargs, color_by=color_by)
                                st.session_state.last_chart = fig

                                st.subheader("Execution Trace")
                                for step in st.session_state.exec_trace:
                                    st.write("- " + step)

                            except Exception as e:
                                st.error(f"Execution error: {e}")
                                st.session_state.messages.append({"role": "assistant", "content": f"Execution error: {e}"})
                    except ValidationError as ve:
                        st.error("Plan validation failed. See details below.")
                        st.code(str(ve))
                        st.session_state.messages.append({"role": "assistant", "content": "Plan validation failed."})
                    except Exception as e:
                        st.error(f"Interpreter error: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"Interpreter error: {e}"})

st.caption("Sort normalization enabled: accepts {'by':'Col','dir':'desc'} or {'Col':'desc'} and standardizes it.")
