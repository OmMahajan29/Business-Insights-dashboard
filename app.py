import os
import time
import json
import streamlit as st
from dotenv import load_dotenv

from core.data import read_csv_preview_cached, read_csv_full, maybe_sample, schema_dataframe
from core.constants import DISPLAY_PREVIEW_ROWS, ROW_LIMIT_HARD, ALLOWED_CHARTS
from core.plan import AnalysisPlan
from core.interpreter import call_interpreter_llm, parse_plan_text
from core.executor import aggregate_df, ExecutionError
from core.viz import choose_chart, render_chart, make_downloads
from core.secrets import get_openrouter_key
from core.telemetry import show_log_viewer_sidebar
from core.share import encode_plan_to_token, decode_token_to_plan, save_named_plan, list_saved_plans

# ---------- Boot ----------
load_dotenv()
OR_KEY = get_openrouter_key()

st.set_page_config(page_title="Business Insights Dashboard", layout="wide")
st.title("LLM-Powered Business Insights Dashboard")

# ---------- Sidebar: Logs ----------
show_log_viewer_sidebar()

# ---------- Session ----------
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

# ---------- Sidebar: Data ----------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Choose data mode", ["CSV (upload)", "SQL (coming soon)"], index=0)

if mode == "CSV (upload)":
    file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    with st.sidebar.expander("CSV read options", expanded=False):
        sep = st.text_input("Delimiter (sep)", value=",")
        header_row = st.number_input("Header row index", min_value=0, value=0, step=1)
        encoding = st.selectbox(
            "Encoding",
            options=[
                "Auto","utf-8","utf-8-sig","cp1252","latin1","iso-8859-1",
                "cp1250","cp932","cp949","cp950","gb18030","utf-16","utf-16-le","utf-16-be"
            ],
            index=0
        )
        nrows_preview = st.number_input("Preview rows", min_value=10, max_value=5000, value=200, step=10)

    if file is not None:
        try:
            _ = read_csv_preview_cached(file, sep, header_row, encoding, int(nrows_preview))
            file.seek(0)
            df_full = read_csv_full(file, sep, header_row, encoding)
            df_full = maybe_sample(df_full)
            st.session_state.df = df_full
            st.session_state.dataset_name = getattr(file, "name", "uploaded.csv")
            st.sidebar.success(f"Loaded: {st.session_state.dataset_name} (rows: {len(df_full):,}, cols: {df_full.shape[1]})")
        except Exception as e:
            st.session_state.df = None
            st.sidebar.error(f"Failed to read CSV: {e}")
else:
    st.sidebar.info("SQL connector coming soon (read-only, SELECT-only).")

# ---------- Sidebar: Chart options ----------
st.sidebar.header("Chart Options")
options = ["Auto"] + sorted(list(ALLOWED_CHARTS))
try:
    default_idx = options.index("Auto")
except ValueError:
    default_idx = 0

chart_override = st.sidebar.selectbox("Chart type (optional override)", options=options, index=default_idx)
color_by = st.sidebar.text_input("Color by (optional dimension)", value="")
download_name = st.sidebar.text_input("Download base name", value="insights_result")

# ---------- Sidebar: Saved analyses ----------
st.sidebar.header("Saved analyses")
saved = list_saved_plans()
names = ["-- select saved --"] + [rec["name"] for rec in saved]
pick = st.sidebar.selectbox("Load a saved plan", names, index=0)
if pick != "-- select saved --":
    rec = next(r for r in saved if r["name"] == pick)
    st.session_state.last_plan = rec["plan"]
    st.sidebar.success(f"Loaded plan: {pick}. Click 'Run saved/URL plan' after loading a dataset.")

new_name = st.sidebar.text_input("Save current plan as (name)", value="")
if st.sidebar.button("Save current plan"):
    if st.session_state.last_plan:
        extras = {"chart_override": chart_override, "color_by": color_by, "download_name": download_name}
        save_named_plan(new_name or f"plan-{int(time.time())}", st.session_state.last_plan, extras)
        st.sidebar.success("Saved. Use the dropdown to reload.")
    else:
        st.sidebar.warning("No plan available to save yet.")

# ---------- URL: load plan if present ----------
if "plan" in st.query_params:
    token = st.query_params["plan"]
    decoded = decode_token_to_plan(token)
    if decoded and decoded.get("plan"):
        st.session_state.last_plan = decoded["plan"]
        extras = decoded.get("extras", {})
        chart_override = extras.get("chart_override", chart_override)
        color_by = extras.get("color_by", color_by)
        download_name = extras.get("download_name", download_name)
        st.info("Plan preloaded from URL. Click 'Run saved/URL plan' to execute after uploading the CSV.")

# ---------- Status ----------
if OR_KEY:
    st.success("LLM status: OpenRouter key loaded (via secrets/env).")
else:
    st.warning("LLM status: No API key found. Interpretation uses the key; execution is local.")

if st.session_state.df is None:
    st.info("Upload a CSV in the sidebar to begin.")
else:
    st.success(f"Dataset ready: {st.session_state.dataset_name}")

# ---------- Dataset overview ----------
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Preview")
    st.dataframe(df.head(DISPLAY_PREVIEW_ROWS), use_container_width=True)
    st.subheader("Schema summary")
    st.dataframe(schema_dataframe(df), use_container_width=True)

# ---------- Chat ----------
st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

col1, col2 = st.columns(2)
with col1:
    run_saved = st.button("Run saved/URL plan")
with col2:
    share_link = st.button("Create shareable link")

prompt = st.chat_input("Ask about your data…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

# Create shareable link for current plan
if share_link:
    if st.session_state.last_plan:
        extras = {"chart_override": chart_override, "color_by": color_by, "download_name": download_name}
        token = encode_plan_to_token(st.session_state.last_plan, extras)
        st.query_params["plan"] = token
        st.success("Share link created. Copy the browser URL to share this exact view.")
    else:
        st.warning("No plan available to share yet. Ask a question or load a saved plan first.")

# Interpret prompt into a plan
if prompt and st.session_state.df is not None:
    if not OR_KEY:
        reply = "Interpreter unavailable (no API key). Add OPENROUTER_API_KEY in .streamlit/secrets.toml or .env."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Interpreting your question into a structured plan..."):
                try:
                    raw = call_interpreter_llm(prompt, st.session_state.df, OR_KEY)
                    plan = parse_plan_text(raw)
                    st.session_state.last_plan = plan.dict()
                    st.success("Analysis plan created.")
                    st.write("Plan (validated):")
                    st.json(plan.dict())
                except Exception as e:
                    st.exception(e)

# Execute saved/URL plan
if run_saved and st.session_state.df is not None and st.session_state.last_plan:
    try:
        plan = AnalysisPlan(**st.session_state.last_plan)
    except Exception as e:
        st.exception(e)
        plan = None

    if plan:
        st.session_state.exec_trace = []
        with st.spinner("Executing plan with safety checks..."):
            try:
                result = aggregate_df(st.session_state.df, plan, st.session_state.exec_trace)
                st.session_state.last_result = result

                # Empty-result guard so the page doesn't look blank
                if result is None or result.empty or len(result.columns) == 0:
                    st.warning("No rows to visualize after applying the plan. Adjust the prompt, remove filters, or pick a numeric metric.")
                else:
                    st.subheader("Result")
                    st.dataframe(result, use_container_width=True)

                    chart, fig_kwargs = choose_chart(plan, result, chart_override)
                    st.subheader(f"Chart ({chart})")
                    fig = render_chart(chart, result, fig_kwargs, color_by=color_by)
                    st.session_state.last_chart = fig

                st.subheader("Execution Trace")
                for step in st.session_state.exec_trace:
                    st.write("- " + step)

                if st.session_state.get("last_chart") is not None and st.session_state.get("last_result") is not None:
                    make_downloads(st.session_state.last_result, st.session_state.last_chart, download_name)

            except ExecutionError as ee:
                st.exception(ee)
            except Exception as e:
                st.exception(e)

st.caption("Fixed control flow: plan -> execute -> result + chart. If nothing draws, an exception or empty-result warning is shown.")
