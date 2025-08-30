from typing import List, Dict, Optional
import pandas as pd
from .plan import AnalysisPlan, FilterSpec
from .constants import ALLOWED_OPS, ALLOWED_AGGS, ROW_LIMIT_HARD

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
        col, val = f.column, f.value
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

    # Build aggregation map for metrics when grouping or aggregating
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
        # No dimensions. Support histogram-style requests (no aggregation table needed).
        if len(mets) == 1 and (plan.chart_suggestion == "hist"):
            # Provide summary stats table for the chosen metric
            col = mets[0]
            result = work[[col]].describe().T.reset_index().rename(columns={"index": "metric"})
            trace.append("Histogram request: plotting distribution from metric column; table shows summary stats.")
        else:
            result = work.agg(agg_map)
            if isinstance(result, pd.Series):
                result = result.to_frame().T
            result.insert(0, "all_rows", "all")
            trace.append(f"Aggregate: {agg_map}")

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
