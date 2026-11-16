"""Recompute a checkpoint's operating points (op_threshs) on the validation split, in place.

    python -m ft.calibrate runs/v1/best.pt

Use after changing operating_points() in model.py; the weights stay the same.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import XrayDataset
from .labels import PATHOLOGIES
from .model import calibrate, device, load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    dev = device()
    val = pd.read_csv(a.labels).query("split == 'val'")
    kw = {"batch_size": 32, "num_workers": a.workers}
    loaders = [DataLoader(XrayDataset(val, "clean"), **kw), DataLoader(XrayDataset(val, "photo", seed=1), **kw)]
    model = load(a.checkpoint).to(dev)
    old = model.op_threshs.cpu().numpy()
    th = calibrate(model, loaders, dev)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    ck["state_dict"]["op_threshs"] = torch.tensor(th, dtype=torch.float32)
    ck["op_threshs"] = th.tolist()
    torch.save(ck, a.checkpoint)
    for name, o, n in zip(PATHOLOGIES, old, th):
        print(f"{name:28s} {o:.4f} -> {n:.4f}" + ("" if np.isclose(o, n) else "  *"))
    print(f"updated {a.checkpoint}")


if __name__ == "__main__":
    main()
