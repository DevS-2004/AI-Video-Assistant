import os
import json
import uuid
from typing import List, Dict, Any, Optional

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history_data.json")

def _sanitize_for_json(data: Any) -> Any:
    """Recursively strip non-serializable objects (like LangChain runnables) for JSON persistence."""
    if isinstance(data, dict):
        clean = {}
        for k, v in data.items():
            if k == "rag_chain":
                continue
            clean[k] = _sanitize_for_json(v)
        return clean
    elif isinstance(data, list):
        return [_sanitize_for_json(x) for x in data]
    elif isinstance(data, (str, int, float, bool)) or data is None:
        return data
    else:
        return str(data)

def load_all_history() -> List[Dict[str, Any]]:
    """Loads all saved session records from disk."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

def save_session(result_payload: Dict[str, Any]) -> str:
    """Saves or updates a session in the persistent history file."""
    history = load_all_history()
    
    session_id = result_payload.get("id") or str(uuid.uuid4())[:8]
    title = result_payload.get("title", "Untitled Session")
    timestamp = result_payload.get("timestamp", "Recent")
    
    clean_result = _sanitize_for_json(result_payload)
    clean_result["id"] = session_id
    
    # Remove existing with same title or id to prevent duplicates
    history = [h for h in history if h.get("id") != session_id and h.get("title") != title]
    
    entry = {
        "id": session_id,
        "title": title,
        "timestamp": timestamp,
        "result": clean_result,
    }
    
    # Prepend newest
    history.insert(0, entry)
    
    # Keep up to 25 sessions
    history = history[:25]
    
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing history file: {e}")
        
    return session_id

def delete_session(session_id: str) -> bool:
    """Deletes a session by ID."""
    history = load_all_history()
    new_hist = [h for h in history if h.get("id") != session_id]
    if len(new_hist) != len(history):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(new_hist, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    return False

def clear_all_history() -> bool:
    """Clears all session history."""
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        return True
    except Exception:
        return False
