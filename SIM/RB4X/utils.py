from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import mujoco
import numpy as np


def load_model(path: Union[str, Path]) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load an MJCF model from `path` and return (model, data).

    Args:
        path: Path to an MJCF XML file (str or Path).

    Returns:
        (model, data): MuJoCo model and freshly-created data with `mj_forward` applied.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {p}")

    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def get_rb4x_handles(
    model: mujoco.MjModel,
    data: Optional[mujoco.MjData] = None,
    *,
    num_regions: int = 12,
    strict: bool = True,
) -> Dict[str, Any]:
    """Resolve RB4X site/mocap IDs (and optional site world positions).

    Args:
        model: MuJoCo model to query for named objects.
        data: Optional MuJoCo data; if provided, includes `xpos` from `data.site_xpos`.
        num_regions: Number of region sites named `region_site{i}`.
        strict: If True, raise on missing names; otherwise keep entries with invalid IDs.

    Returns:
        Dict with keys:
          - "feet":    {1..4: {"site_id": int, "xpos": np.ndarray|None}}
          - "regions": {1..N: {"site_id": int, "xpos": np.ndarray|None}}
          - "anchors": {1..4: mocap_id (int)}
    """
    def require_id(obj: mujoco.mjtObj, name: str) -> int:
        idx = mujoco.mj_name2id(model, obj, name)
        if idx < 0 and strict:
            raise KeyError(f"Missing {obj.name.lower()} '{name}'")
        return int(idx)

    def site_entry(name: str) -> Dict[str, Any]:
        sid = require_id(mujoco.mjtObj.mjOBJ_SITE, name)
        entry: Dict[str, Any] = {"site_id": sid, "xpos": None}
        if sid >= 0 and data is not None:
            entry["xpos"] = np.asarray(data.site_xpos[sid], dtype=float).copy()
        return entry

    feet: Dict[int, Dict[str, Any]] = {k: site_entry(f"boomEndSite{k}") for k in range(1, 5)}
    regions: Dict[int, Dict[str, Any]] = {
        i: site_entry(f"region_site{i}") for i in range(1, num_regions + 1)
    }

    anchors: Dict[int, int] = {}
    for k in range(1, 5):
        body_name = f"foot{k}_anchor_body"
        bid = require_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        mocap_id = int(model.body_mocapid[bid])
        if mocap_id < 0:
            if strict:
                raise ValueError(f"Body '{body_name}' is not mocap='true'")
            continue
        anchors[k] = mocap_id

    return {"feet": feet, "regions": regions, "anchors": anchors}
