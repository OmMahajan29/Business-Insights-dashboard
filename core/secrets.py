import os
import streamlit as st
from streamlit.runtime.secrets import StreamlitSecretNotFoundError

def get_openrouter_key() -> str | None:
    # Try Streamlit secrets first; if not configured, fall back to environment (.env)
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except StreamlitSecretNotFoundError:
        pass
    return os.getenv("OPENROUTER_API_KEY")
