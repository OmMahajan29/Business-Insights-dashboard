import io
from typing import Dict, Optional, Sequence, Tuple
import pandas as pd
import streamlit as st
from .constants import SAMPLE_SIZE, SAMPLE_THRESHOLD

# Try to import charset detection (optional)
try:
    import charset_normalizer as cdet  # pip install charset-normalizer
    HAVE_DETECT = True
except Exception:
    HAVE_DETECT = False

COMMON_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "cp1252",
    "latin1",          # iso-8859-1; accepts any byte [2]
    "iso-8859-1",
    "cp1250",
    "cp932",           # Japanese
    "cp949",           # Korean
    "cp950",           # Traditional Chinese
    "gb18030",         # Simplified Chinese [11]
    "utf-16",
    "utf-16-le",
    "utf-16-be",
]

def _detect_encoding(raw_bytes: bytes) -> Optional[str]:
    if not HAVE_DETECT or not raw_bytes:
        return None
    res = cdet.from_bytes(raw_bytes).best()
    if res and res.encoding:
        return res.encoding
    return None

def _try_read_bytes(raw_bytes: bytes, sep, header, encoding: Optional[str], nrows: Optional[int] = None):
    """
    Attempt to read CSV from bytes with a specific encoding, using backslashreplace to avoid hard failures. [3]
    """
    errors = "backslashreplace"  # avoid hard stops; preserve bytes visibly [3]
    return pd.read_csv(io.BytesIO(raw_bytes), sep=sep, header=header, encoding=encoding or "utf-8", encoding_errors=errors, nrows=nrows)

@st.cache_data(show_spinner=False)
def read_csv_preview_cached(_file, _sep, _header, _enc, _nrows: int):
    raw = _file.getvalue()
    if _enc and _enc.lower() != "auto":
        return _try_read_bytes(raw, _sep, _header, _enc, nrows=int(_nrows))
    enc = _detect_encoding(raw) or "utf-8"
    try:
        return _try_read_bytes(raw, _sep, _header, enc, nrows=int(_nrows))
    except Exception:
        # fallback ladder across common encodings [2][6]
        for e in COMMON_ENCODINGS:
            try:
                return _try_read_bytes(raw, _sep, _header, e, nrows=int(_nrows))
            except Exception:
                continue
        # last resort: latin1 never fails but may yield mojibake [2]
        return _try_read_bytes(raw, _sep, _header, "latin1", nrows=int(_nrows))

def read_csv_full(_file, _sep, _header, _enc):
    raw = _file.getvalue()
    if _enc and _enc.lower() != "auto":
        return _try_read_bytes(raw, _sep, _header, _enc)
    enc = _detect_encoding(raw) or "utf-8"
    try:
        return _try_read_bytes(raw, _sep, _header, enc)
    except Exception:
        for e in COMMON_ENCODINGS:
            try:
                return _try_read_bytes(raw, _sep, _header, e)
            except Exception:
                continue
        return _try_read_bytes(raw, _sep, _header, "latin1")

def maybe_sample(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > SAMPLE_THRESHOLD:
        return df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    return df

def schema_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    dtypes = df.dtypes.astype(str)
    non_null_counts = df.notna().sum()
    null_counts = df.isna().sum()
    sample_vals = df.head(3).astype(str)
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": dtypes[col],
            "non_null": int(non_null_counts[col]),
            "nulls": int(null_counts[col]),
            "unique": int(df[col].nunique(dropna=True)),
            "example_values": ", ".join(sample_vals[col].tolist())
        })
    return pd.DataFrame(rows)
