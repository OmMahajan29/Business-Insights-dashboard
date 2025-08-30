import json
import time
import hashlib
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "llm_cache.jsonl"
DEFAULT_TTL_SEC = 60 * 60 * 24  # 24 hours

def _hash_key(model: str, schema_text: str, question: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update(schema_text.encode("utf-8"))
    h.update(b"|")
    h.update(question.encode("utf-8"))
    return h.hexdigest()

def get_cached(model: str, schema_text: str, question: str) -> Optional[str]:
    key = _hash_key(model, schema_text, question)
    if not CACHE_FILE.exists():
        return None
    with CACHE_FILE.open("r", encoding="utf-8") as f:
        for line in reversed(f.readlines()):
            try:
                rec = json.loads(line)
                if rec.get("key") == key and time.time() - rec.get("ts", 0) < rec.get("ttl", DEFAULT_TTL_SEC):
                    return rec.get("value")
            except Exception:
                continue
    return None

def set_cached(model: str, schema_text: str, question: str, value: str, ttl: int = DEFAULT_TTL_SEC) -> None:
    key = _hash_key(model, schema_text, question)
    rec = {"key": key, "ts": time.time(), "ttl": ttl, "value": value}
    with CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
