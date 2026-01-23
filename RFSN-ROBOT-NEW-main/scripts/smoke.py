from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import mujoco

# Allow running as a plain script: `python scripts/smoke.py ...`
# In that case, Python sets sys.path[0] to ./scripts, so the repo root
# (which contains the `rfsn/` package) isn't importable unless we add it.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rfsn.harness_v2 import RFSNHarnessV2


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return {k: _to_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="panda_table_cube.xml")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", type=str, default="runs/smoke")
    ap.add_argument("--mode", type=str, default="rfsn", choices=["mpc_only", "rfsn", "rfsn_learning"])
    ap.add_argument("--controller", type=str, default="joint_mpc", choices=["pd", "joint_mpc", "task_mpc", "impedance"])
    args = ap.parse_args()

    # determinism
    random.seed(args.seed)
    np.random.seed(args.seed)

    # headless-friendly defaults (harmless elsewhere)
    os.environ.setdefault("MUJOCO_GL", "egl")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # hard reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # harness (V12 entry point)
    harness = RFSNHarnessV2(
        model=model,
        data=data,
        mode=args.mode,
        controller_mode=args.controller,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_jsonl = outdir / f"smoke_{stamp}.jsonl"

    bad_nan = 0
    bad_self_collision = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i in range(args.steps):
            obs = harness.step()

            if np.isnan(obs.q).any() or np.isnan(obs.qd).any():
                bad_nan += 1
            if getattr(obs, "self_collision", False):
                bad_self_collision += 1

            rec = {
                "i": i,
                "t": float(obs.t),
                "q": obs.q.copy(),
                "qd": obs.qd.copy(),
                "self_collision": bool(getattr(obs, "self_collision", False)),
                "table_collision": bool(getattr(obs, "table_collision", False)),
                "penetration": float(getattr(obs, "penetration", 0.0)),
            }
            f.write(json.dumps(_to_jsonable(rec)) + "\n")

    summary = {
        "steps": args.steps,
        "seed": args.seed,
        "model": str(model_path),
        "bad_nan": bad_nan,
        "bad_self_collision": bad_self_collision,
        "log": str(out_jsonl),
    }
    print(json.dumps(summary, indent=2))

    if bad_nan:
        raise SystemExit(f"FAIL: NaNs detected in {bad_nan} steps")
    if bad_self_collision:
        raise SystemExit(f"FAIL: self-collision flagged in {bad_self_collision} steps")

    print("PASS: smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
