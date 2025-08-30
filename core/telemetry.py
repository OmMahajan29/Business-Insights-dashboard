import json
import re
from pathlib import Path
import streamlit as st

LOG_DIR = Path(".cache")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "prompt_logs.jsonl"

REDACTION_PATTERNS = [
    re.compile(r"\bsk-[a-zA-Z0-9\-_=]{10,}\b"),
    re.compile(r"\bppx-[a-zA-Z0-9\-_=]{10,}\b"),
    re.compile(r"\b\d{12,19}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]

def redact(text: str) -> str:
    out = text
    for pat in REDACTION_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out

def log_prompt(event: dict) -> None:
    event = {k: (redact(v) if isinstance(v, str) else v) for k, v in event.items()}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

def show_log_viewer_sidebar():
    st.sidebar.header("Logs")
    if st.sidebar.button("Refresh logs"):
        st.experimental_rerun()
    if LOG_FILE.exists():
        with LOG_FILE.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-200:]
        if lines:
            st.sidebar.caption("Recent prompt events (redacted):")
            for ln in reversed(lines):
                try:
                    st.sidebar.code(ln.strip(), language="json")
                except Exception:
                    continue
    else:
        st.sidebar.caption("No logs yet.")
