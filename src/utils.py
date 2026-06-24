from typing import Dict, Any, List
import json
import os



def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# LEDGER UPDATE
# =========================================================

def apply_ledger_updates(agent_ledger: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    ledger = dict(agent_ledger)

    for path, value in updates.items():
        set_path(ledger, path, value)

    return ledger


def set_path(data: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = data

    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]

    cur[keys[-1]] = value


def get_path(data: Dict[str, Any], path: str):
    """safe nested getter"""
    try:
        keys = path.split(".")
        cur = data

        for k in keys:
            cur = cur[k]
        return cur
    except Exception:
        return None



def apply_effect(state: Dict[str, Any], effect: Dict[str, Any]) -> Dict[str, Any]:

    for key, value in effect.items():

        if isinstance(value, str) and value.lstrip("+-").replace(".", "", 1).isdigit():
            delta = float(value)
            old = get_path(state, key)
            if old is None:
                old = 0
            set_path(state, key, old + delta)

        else:
            set_path(state, key, value)

    return state


# =========================================================
# UTILITIES FOR EVALUATION
# =========================================================

def values_equal(a: Any, b: Any) -> bool:
    """robust equality check"""
    return a == b


def contains_required_actions(planned_actions: List[str], required_actions: List[str]) -> bool:
    """
    check if all required actions are contained in planned actions
    """

    return all(
        any(req in act for act in planned_actions)
        for req in required_actions
    )