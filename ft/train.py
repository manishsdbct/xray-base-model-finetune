"""Fine-tune TorchXRayVision's DenseNet-121 (densenet121-res224-all) on data/labels.csv.

    python -m ft.train --out runs/first

Every batch mixes clean scans with simulated phone photos (--p-photo). Labels with fewer than
--min-positives positives in train are not trained directly; for those the model is held to the
base model's answer on the clean image (--keep), so it does not forget what it knew. That hold also
teaches it to give the same answer for a photo as for the clean scan.

Writes runs/<name>/best.pt (weights plus new operating points, ready for the app) and metrics.json.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import XrayDataset
from .labels import PATHOLOGIES
from .model import aucs, base_model, calibrate, device, logits


class Pairs(Dataset):
    """(augmented image, clean image, labels): the clean one is for the base model's answer."""

    def __init__(self, table, p_photo):
        self.aug = XrayDataset(table, "train", p_photo)
        self.clean = XrayDataset(table, "clean")

    def __len__(self):
        return len(self.aug)

    def __getitem__(self, i):
        x, y = self.aug[i]
        return x, self.clean[i][0], y


def masked_bce(z: torch.Tensor, y: torch.Tensor, use: torch.Tensor) -> torch.Tensor:
    """BCE over labels in `use` that are not doubtful (NaN)."""
    ok = ~torch.isnan(y) & use
    if not ok.any():
        return z.sum() * 0
    return F.binary_cross_entropy_with_logits(z[ok], y[ok])


def evaluate(model, loaders, dev, trained) -> dict:
    out = {}
    for name, loader in loaders.items():
        z, y = logits(model, loader, dev)
        a = aucs(z, y)
        got = [a[k] for k in trained if k in a]
        out[name] = {"auc": a, "mean_auc_trained": float(np.mean(got)) if got else float("nan")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate for the backbone")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="learning rate for the last layer")
    ap.add_argument("--freeze-epochs", type=int, default=1, help="epochs that train only the last layer")
    ap.add_argument("--p-photo", type=float, default=0.5, help="share of training images made into phone photos")
    ap.add_argument("--min-positives", type=int, default=40)
    ap.add_argument("--keep", type=float, default=1.0, help="weight of holding untrained labels to the base model")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only this many images per split (quick check)")
    a = ap.parse_args()

    torch.manual_seed(0)
    dev = device()
    t = pd.read_csv(a.labels)
    split = {s: t[t.split == s] for s in ("train", "val")}
    if a.limit:
        split = {s: d.sample(min(a.limit, len(d)), random_state=0) for s, d in split.items()}
    pos = (split["train"][PATHOLOGIES] == 1).sum()
    trained = [p for p in PATHOLOGIES if pos[p] >= a.min_positives]
    use = torch.tensor([p in trained for p in PATHOLOGIES], device=dev)
    print(f"device {dev}; {len(split['train'])} train, {len(split['val'])} val images")
    print(f"trained labels ({len(trained)}): {', '.join(trained)}")
    print(f"held to base model: {', '.join(p for p in PATHOLOGIES if p not in trained)}")

    kw = {"batch_size": a.batch, "num_workers": a.workers, "persistent_workers": a.workers > 0}
    train = DataLoader(Pairs(split["train"], a.p_photo), shuffle=True, drop_last=True, **kw)
    val = {
        "val_clean": DataLoader(XrayDataset(split["val"], "clean"), **kw),
        "val_photo": DataLoader(XrayDataset(split["val"], "photo", seed=1), **kw),
    }

    model = base_model().to(dev)
    model.op_threshs = None  # train on raw scores
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    head = list(model.classifier.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=1e-4)
    steps = a.epochs * len(train)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=steps, pct_start=0.1)

    a.out.mkdir(parents=True, exist_ok=True)
    history = [{"epoch": 0, **evaluate(model, val, dev, trained)}]
    score0 = np.nanmean([history[0][k]["mean_auc_trained"] for k in val])
    print(f"base model: val clean {history[0]['val_clean']['mean_auc_trained']:.3f}, "
          f"photo {history[0]['val_photo']['mean_auc_trained']:.3f}")  # fmt: skip
    best = -1.0

    for epoch in range(1, a.epochs + 1):
        frozen = epoch <= a.freeze_epochs
        for p in body:
            p.requires_grad = not frozen
        model.train()
        if frozen:
            model.features.eval()  # keep batch-norm statistics while only the head trains
        start, total, n = time.time(), 0.0, 0
        for x, clean, y in train:
            x, clean, y = x.to(dev), clean.to(dev), y.to(dev)
            z = model(x)
            loss = masked_bce(z, y, use)
            if a.keep:
                with torch.no_grad():
                    target = torch.sigmoid(teacher(clean))
                loss = loss + a.keep * F.binary_cross_entropy_with_logits(z[:, ~use], target[:, ~use])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        model.eval()
        r = {"epoch": epoch, "loss": total / n, **evaluate(model, val, dev, trained)}
        history.append(r)
        score = np.nanmean([r[k]["mean_auc_trained"] for k in val])
        mark = ""
        if epoch == 1 or score > best:  # NaN only when val is too small to score (--limit)
            best, mark = score, "  <- best"
            torch.save({"state_dict": model.state_dict()}, a.out / "last_best_weights.pt")
        print(f"epoch {epoch}: loss {r['loss']:.4f}, val clean {r['val_clean']['mean_auc_trained']:.3f}, "
              f"photo {r['val_photo']['mean_auc_trained']:.3f} ({time.time() - start:.0f}s){mark}")  # fmt: skip

    # New operating points from the best weights, on clean and photo validation images together, so
    # a score of 0.5 keeps meaning "at the operating point" in the app.
    model.load_state_dict(torch.load(a.out / "last_best_weights.pt", map_location=dev)["state_dict"])
    th = calibrate(model, val.values(), dev)
    model.op_threshs = torch.tensor(th, dtype=torch.float32, device=dev)  # saved in state_dict
    torch.save(
        {"state_dict": model.state_dict(), "op_threshs": th.tolist(), "pathologies": PATHOLOGIES,
         "base_weights": "densenet121-res224-all", "trained_labels": trained, "args": vars(a) | {
             "labels": str(a.labels), "out": str(a.out)}},
        a.out / "best.pt",
    )  # fmt: skip
    (a.out / "last_best_weights.pt").unlink()
    json.dump(
        {"trained_labels": trained, "base_score": score0, "best_score": best, "history": history},
        open(a.out / "metrics.json", "w"), indent=2,
    )  # fmt: skip
    print(f"saved {a.out / 'best.pt'} (val score {score0:.3f} -> {best:.3f})")


if __name__ == "__main__":
    main()
