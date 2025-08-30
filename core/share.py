import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

STORE = Path(".cache")
STORE.mkdir(exist_ok=True)
PLANS_FILE = STORE / "saved_plans.jsonl"

def _b64_url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

def _b64_url_decode(s: str) -> bytes:
    padding = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + padding)

def encode_plan_to_token(plan_dict: Dict[str, Any], extras: Dict[str, Any]) -> str:
    payload = {"plan": plan_dict, "extras": extras}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _b64_url_encode(raw)

def decode_token_to_plan(token: str) -> Optional[Dict[str, Any]]:
    try:
        raw = _b64_url_decode(token)
        obj = json.loads(raw.decode("utf-8"))
        return obj
    except Exception:
        return None

def save_named_plan(name: str, plan_dict: Dict[str, Any], extras: Dict[str, Any]) -> None:
    rec = {"name": name, "plan": plan_dict, "extras": extras}
    with PLANS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def list_saved_plans() -> List[Dict[str, Any]]:
    if not PLANS_FILE.exists():
        return []
    items = []
    with PLANS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    # latest first
    items.reverse()
    return items
