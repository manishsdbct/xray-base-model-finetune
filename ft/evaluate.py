"""Compare the base model with fine-tuned checkpoints on the test split.

    python -m ft.evaluate runs/first/best.pt
    python -m ft.evaluate runs/first/best.pt --cases ~/Documents/xray-saathi/data/eval

Three test sets: the test split as stored (clean), the same images made into simulated phone
photos (photo), and optionally folders of real cases with scan.jpg + report.txt (cases).
Reports AUC per label, and what the app sees: with its rule "score >= 0.5 is a hint", how many
present findings get a hint (caught) and how many absent ones do (false hints).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CaseDataset, XrayDataset
from .labels import PATHOLOGIES
from .model import aucs, device, load, logits

APP_MIN_SCORE = 0.5  # backend/app/pipeline/xray_model.py MIN_SCORE (--min-score)


@torch.no_grad()
def app_scores(model, loader, dev) -> np.ndarray:
    """Scores exactly as the app gets them (sigmoid, then the operating-point scaling)."""
    return np.concatenate([model(x.to(dev)).float().cpu().numpy() for x, _ in loader])


def hint_rates(scores: np.ndarray, y: np.ndarray, min_score: float = APP_MIN_SCORE) -> tuple[float, float]:
    ok = ~np.isnan(y)
    hit = scores >= min_score
    pos, neg = ok & (y == 1), ok & (y == 0)
    return hit[pos].mean() if pos.any() else np.nan, hit[neg].mean() if neg.any() else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="*", help="fine-tuned best.pt files (the base model is always included)")
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--cases", type=Path, nargs="*", default=[], help="folders of cases with scan.jpg + report.txt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=APP_MIN_SCORE, help="the app's hint cut-off")
    a = ap.parse_args()

    dev = device()
    t = pd.read_csv(a.labels)
    test = t[t.split == "test"]
    kw = {"batch_size": 32, "num_workers": a.workers}
    sets = {
        "clean": DataLoader(XrayDataset(test, "clean"), **kw),
        "photo": DataLoader(XrayDataset(test, "photo", seed=2), **kw),
    }
    for c in a.cases:
        sets[f"cases:{c.name}"] = DataLoader(CaseDataset(c), **kw)
    print(f"test: {len(test)} images" + "".join(f"; {k}: {len(v.dataset)}" for k, v in sets.items() if ":" in k))

    names = ["base"] + [str(c.parent.name) for c in a.checkpoints]
    rows, per_label = [], {}
    for name, ck in zip(names, [None, *a.checkpoints]):
        model = load(ck).to(dev)
        for sname, loader in sets.items():
            z, y = logits(model, loader, dev)
            au = aucs(z, y, min_pos=3 if ":" in sname else 5)
            caught, false = hint_rates(app_scores(model, loader, dev), y, a.min_score)
            rows.append({"model": name, "set": sname, "labels": len(au), "mean AUC": np.mean(list(au.values())) if au else np.nan,
                         "caught": caught, "false hints": false})  # fmt: skip
            per_label[(name, sname)] = au

    print("\n" + pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))
    print("\nAUC per label (labels with enough positives in that set):")
    table = pd.DataFrame(per_label).reindex([p for p in PATHOLOGIES if any(p in v for v in per_label.values())])
    print(table.to_string(float_format="%.3f", na_rep="-"))


if __name__ == "__main__":
    main()
