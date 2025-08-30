import streamlit as st
from typing import Dict, Any

PREFS_KEY = "user_prefs"

DEFAULT_PREFS: Dict[str, Any] = {
    "sep": ",",
    "header_row": 0,
    "encoding": "utf-8",
    "nrows_preview": 200,
    "chart_override": "Auto",
    "color_by": "",
    "download_name": "insights_result",
}

def _ensure():
    if PREFS_KEY not in st.session_state:
        st.session_state[PREFS_KEY] = DEFAULT_PREFS.copy()

def get_prefs() -> Dict[str, Any]:
    _ensure()
    return st.session_state[PREFS_KEY]

def set_pref(key: str, value):
    _ensure()
    st.session_state[PREFS_KEY][key] = value
