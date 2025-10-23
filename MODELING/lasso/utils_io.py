# utils_io.py
from pathlib import Path
import numpy as np
import json

def save_solution(filename: str, X_opt: np.ndarray, U_opt: np.ndarray, meta: dict | None = None):
    """
    Save optimal trajectory to an .npz next to caller script.
    Also writes a small JSON sidecar with metadata for readability.
    """
    p = Path(filename).with_suffix(".npz")
    np.savez_compressed(p, X_opt=X_opt, U_opt=U_opt, meta=(meta or {}))
    # Optional readable sidecar (nice for quick inspection)
    if meta:
        p_json = p.with_suffix(".json")
        # Make everything JSON-friendly
        clean = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in meta.items()}
        with open(p_json, "w") as f:
            json.dump(clean, f, indent=2)
    print(f"[utils_io] Saved: {p.name}" + (f" (+ {p.with_suffix('.json').name})" if meta else ""))

def load_solution(filename: str):
    """
    Load previously saved optimal trajectory (.npz).
    Returns X_opt, U_opt, meta (dict).
    """
    p = Path(filename).with_suffix(".npz")
    data = np.load(p, allow_pickle=True)
    X_opt = data["X_opt"]
    U_opt = data["U_opt"]
    meta = data["meta"].item() if "meta" in data else {}
    return X_opt, U_opt, meta
