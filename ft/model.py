"""Loading the base and fine-tuned models, scoring, and choosing operating points."""

from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from sklearn.metrics import roc_auc_score, roc_curve

from .labels import PATHOLOGIES

BASE_WEIGHTS = "densenet121-res224-all"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def base_model() -> xrv.models.DenseNet:
    m = xrv.models.DenseNet(weights=BASE_WEIGHTS)
    assert list(m.pathologies) == PATHOLOGIES, m.pathologies
    return m


def load(path: Path | None) -> xrv.models.DenseNet:
    """The base model, or a fine-tuned checkpoint on top of it. Both give calibrated scores where
    0.5 is the operating point, as the app expects."""
    m = base_model()
    if path:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["state_dict"])  # includes op_threshs, the new operating points
    return m.eval()


@torch.no_grad()
def logits(model, loader, dev) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores (before the sigmoid) and labels for a whole loader."""
    saved = model.op_threshs
    model.op_threshs = None  # forward() then returns raw scores
    out, ys = [], []
    try:
        for x, y in loader:
            out.append(model(x.to(dev)).float().cpu().numpy())
            ys.append(y.numpy())
    finally:
        model.op_threshs = saved
    return np.concatenate(out), np.concatenate(ys)


def aucs(scores: np.ndarray, y: np.ndarray, min_pos: int = 5) -> dict[str, float]:
    """AUC per label that has at least min_pos positives and some negatives; doubtful labels skipped."""
    out = {}
    for j, name in enumerate(PATHOLOGIES):
        ok = ~np.isnan(y[:, j])
        t, s = y[ok, j], scores[ok, j]
        if t.sum() >= min_pos and (t == 0).sum() >= min_pos:
            out[name] = float(roc_auc_score(t, s))
    return out


def operating_points(probs: np.ndarray, y: np.ndarray, min_pos: int = 10, max_fpr: float = 0.02) -> np.ndarray:
    """Per label, the probability at which a score counts as a hint (0.5 after op_norm).

    With at least min_pos positives on the validation set: the point that best separates present
    from absent (Youden's J). With fewer, sensitivity cannot be measured, so the point is set so
    that only max_fpr of scans without the finding reach it; otherwise such labels hover at the
    cut-off on every photo and add noise."""
    th = np.zeros(len(PATHOLOGIES))
    for j in range(len(PATHOLOGIES)):
        ok = ~np.isnan(y[:, j])
        t, p = y[ok, j], probs[ok, j]
        if t.sum() >= min_pos and (t == 0).sum() >= min_pos:
            fpr, tpr, cut = roc_curve(t, p)
            th[j] = cut[np.argmax(tpr - fpr)]
        else:
            th[j] = np.quantile(p[t == 0], 1 - max_fpr)
    return np.clip(th, 1e-4, 1 - 1e-4)


def calibrate(model, loaders, dev) -> np.ndarray:
    """Operating points from the validation loaders (clean and photo together)."""
    zs, ys = zip(*(logits(model, loader, dev) for loader in loaders))
    return operating_points(1 / (1 + np.exp(-np.concatenate(zs))), np.concatenate(ys))
