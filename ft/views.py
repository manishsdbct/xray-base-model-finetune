"""Frontal or lateral, decided from the image, since the dataset's projection column is often wrong
(for example CXR3198_IM-1509-3001.png is a lateral marked Frontal). Uses TorchXRayVision's view
classifier and caches the answers in data/views.csv."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from tqdm import tqdm

from .model import device


def _tensor(path: str) -> torch.Tensor:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = xrv.datasets.normalize(gray.astype(np.float32), 255)
    return torch.from_numpy(cv2.resize(img, (224, 224))[None, ...])


@torch.no_grad()
def predict(paths: list[str], cache: Path = Path("data/views.csv")) -> dict[str, float]:
    """Probability that each image is frontal."""
    done = pd.read_csv(cache).set_index("path").p_frontal.to_dict() if cache.exists() else {}
    todo = [p for p in paths if p not in done]
    if todo:
        dev = device()
        model = xrv.baseline_models.xinario.ViewModel().to(dev).eval()
        for i in tqdm(range(0, len(todo), 64), desc="views"):
            batch = todo[i : i + 64]
            x = torch.stack([_tensor(p) for p in batch]).to(dev)
            p = torch.softmax(model(x), dim=1)[:, 0].float().cpu().numpy()
            done.update(zip(batch, p.tolist()))
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"path": list(done), "p_frontal": list(done.values())}).to_csv(cache, index=False)
    return {p: done[p] for p in paths}
