import json
import requests
import hashlib
from .plan import AnalysisPlan, dataset_schema_summary, SYSTEM_PROMPT, normalize_sort

# Simple cache/telemetry remain optional; remove imports if not using them
try:
    from .cache import get_cached, set_cached
except Exception:
    def get_cached(*a, **k): return None
    def set_cached(*a, **k): return None

VALID_AGGS = {"sum","avg","count","min","max"}
MODEL_NAME = "mistralai/mistral-7b-instruct:free"

def _sanitize_metric_aggs(obj):
    aggs = obj.get("metric_aggs") or {}
    cleaned = {}
    for k, v in aggs.items():
        vv = str(v).lower().strip()
        cleaned[k] = vv if vv in VALID_AGGS else "avg"
    obj["metric_aggs"] = cleaned

def parse_plan_text(raw: str) -> AnalysisPlan:
    s = raw.strip()
    lines = s.splitlines()
    if lines:
        first = lines[0].strip().lower()
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
    obj = json.loads(s)
    obj["sort"] = normalize_sort(obj.get("sort"))
    _sanitize_metric_aggs(obj)
    return AnalysisPlan(**obj)

def call_interpreter_llm(user_question: str, df, api_key: str) -> str:
    schema_text = dataset_schema_summary(df)
    schema_hash = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()[:12]

    cached = get_cached(MODEL_NAME, schema_text, user_question)
    if cached is not None:
        return cached

    base_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Business Insights Dashboard",
    }
    user_prompt = (
        f"Dataset schema: {schema_text}\n\n"
        f"User question: {user_question}\n\n"
        "Return only the JSON."
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 600,
    }

    resp = requests.post(base_url, headers=headers, json=payload, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        # Surface server message for debugging in terminal
        print("OPENROUTER ERROR:", resp.status_code, resp.text)
        raise
    data = resp.json()
    out = data["choices"][0]["message"]["content"]
    set_cached(MODEL_NAME, schema_text, user_question, out)
    return out
