from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
from nuscenes.nuscenes import NuScenes

from dataloader import DoScenesNuScenesDataset, FUTURE_LEN, load_paths


def world_to_local(points_xy: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """Convert world xy points to anchor-local frame (forward = +x)."""
    c = float(np.cos(anchor_yaw))
    s = float(np.sin(anchor_yaw))
    rel = points_xy - anchor_xy[None, :]
    # R(-yaw) @ rel
    x_local = c * rel[:, 0] + s * rel[:, 1]
    y_local = -s * rel[:, 0] + c * rel[:, 1]
    return np.stack([x_local, y_local], axis=1)


def predict_constant_velocity(history_xy_local: np.ndarray, history_t_us: np.ndarray) -> np.ndarray:
    """12-step prediction from last observed velocity in local frame."""
    if len(history_xy_local) < 2 or len(history_t_us) < 2:
        v = np.zeros(2, dtype=np.float64)
        dt = 0.5
    else:
        p_last = history_xy_local[-1]
        p_prev = history_xy_local[-2]
        dt_last = float(history_t_us[-1] - history_t_us[-2]) / 1e6
        if dt_last <= 1e-6:
            dt_last = 0.5
        v = (p_last - p_prev) / dt_last

        dt = dt_last

    # advance from anchor at origin. Estimate anchor velocity from last history sample.
    p_hist_last = history_xy_local[-1] if len(history_xy_local) else np.zeros(2, dtype=np.float64)
    dt_anchor = float(dt)
    if dt_anchor <= 1e-6:
        dt_anchor = 0.5
    v_anchor = (np.zeros(2, dtype=np.float64) - p_hist_last) / dt_anchor

    preds: List[np.ndarray] = []
    for k in range(FUTURE_LEN):
        t = (k + 1) * dt
        preds.append(v_anchor * t)
    return np.asarray(preds, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build doScenes Track-2 CSV submission (constant-velocity baseline).")
    parser.add_argument("--paths", type=str, default="paths.txt", help="Path to paths.txt")
    parser.add_argument("--version", type=str, default="v1.0-test", help="nuScenes version (default: v1.0-test)")
    parser.add_argument("--output", type=str, default="submissions/submission_track2_cv.csv", help="Output CSV path")
    args = parser.parse_args()

    nusc_root, annotations_path = load_paths(args.paths)

    nusc = NuScenes(version=args.version, dataroot=nusc_root, verbose=True)
    dataset = DoScenesNuScenesDataset(
        nusc=nusc,
        annotations=annotations_path,
        camera_channels=("CAM_FRONT",),
        include_blank_instructions=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["sample_token"] + [f"{axis}{i}" for i in range(1, FUTURE_LEN + 1) for axis in ("x", "y")]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for idx in range(len(dataset)):
            item = dataset[idx]
            anchor_token = item["anchor_sample_token"]

            history_xy_world = item["history_xy"].numpy()
            history_t_us = item["history_timestamps_us"].numpy()
            anchor_xy = item["anchor_xy"].numpy()
            anchor_yaw = float(item["anchor_yaw"].item())

            history_xy_local = world_to_local(history_xy_world, anchor_xy, anchor_yaw)
            pred_local = predict_constant_velocity(history_xy_local, history_t_us)

            row = [anchor_token]
            for p in pred_local:
                row.extend([f"{float(p[0]):.6f}", f"{float(p[1]):.6f}"])
            writer.writerow(row)

    print(f"Wrote {out_path} with {len(dataset)} rows")


if __name__ == "__main__":
    main()
