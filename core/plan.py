from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

AggType = Literal["sum", "avg", "count", "min", "max"]
SortDir = Literal["asc", "desc"]

class FilterSpec(BaseModel):
    column: str
    op: Literal["=", "!=", ">", "<", ">=", "<=", "in", "not in", "contains", "startswith", "endswith"]
    value: Any

class AnalysisPlan(BaseModel):
    intent: str = Field(..., description="User intent.")
    metrics: List[str] = Field(default_factory=list)
    metric_aggs: Dict[str, AggType] = Field(default_factory=dict)
    dimensions: List[str] = Field(default_factory=list)
    timeframe: Optional[str] = None
    sort: Optional[Dict[str, SortDir]] = Field(default=None, description='e.g., {"Sales":"desc"}')
    filters: List[FilterSpec] = Field(default_factory=list)
    limit: Optional[int] = 20
    chart_suggestion: Optional[str] = None
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    notes: Optional[str] = None

def dataset_schema_summary(df) -> str:
    cols = [f"{c} ({t})" for c, t in zip(df.columns, df.dtypes.astype(str).tolist())]
    return ", ".join(cols)

SYSTEM_PROMPT = (
    "You are a data analysis planner. Output STRICT JSON for AnalysisPlan with these fields. "
    "For 'sort', ALWAYS use a single-key object mapping COLUMN to DIRECTION, e.g. {\"Sales\":\"desc\"}. "
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

def normalize_sort(obj) -> Optional[Dict[str, str]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        if len(obj) == 1:
            k, v = list(obj.items())[0]
            vv = str(v).lower()
            if vv in {"asc", "desc"}:
                return {k: vv}
        by = obj.get("by") or obj.get("column")
        dir_ = obj.get("dir") or obj.get("direction")
        if by and isinstance(dir_, str) and dir_.lower() in {"asc", "desc"}:
            return {by: dir_.lower()}
    return None
