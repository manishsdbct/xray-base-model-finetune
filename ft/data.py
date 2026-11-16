"""The IU X-ray images, their labels, the train/val/test split, and the phone-photo augmentation.

Images go through the same steps as in the ScanArth app (backend/app/pipeline/image_prep.py and
xray_model.py): grayscale, levels stretch + CLAHE, squash to 224x224, TorchXRayVision scaling.
Training on anything else would tune the model for images it never gets.
"""

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchxrayvision as xrv
from torch.utils.data import Dataset

from .labels import PATHOLOGIES, label

SIZE = 224


def study_of(image_id: str) -> str:
    return image_id.split("_IM")[0]  # CXR3468_IM-1684-0001-0001.png -> CXR3468


def split_of(study: str, val: int = 10, test: int = 10) -> str:
    """Fixed split by study, so a patient's images never land on both sides."""
    n = int(hashlib.sha1(study.encode()).hexdigest(), 16) % 100
    return "test" if n < test else "val" if n < test + val else "train"


def app_eval_studies(dirs: list[Path]) -> set[str]:
    """Studies used by the app's eval sets (each case has a source.txt). They always go to test, so
    the app's eval stays a fair test of the fine-tuned model."""
    out = set()
    for d in dirs:
        for src in d.glob("*/source.txt"):
            m = re.search(r"\b(?:study CXR|uid )(\d+)", src.read_text())
            if m:
                out.add(f"CXR{m.group(1)}")
    return out


def load_table(csv: Path, images: Path, eval_dirs: list[Path]) -> pd.DataFrame:
    """One row per study: its most clearly frontal image, path, split and the 18 labels. The view
    comes from the image (views.py), not the dataset's projection column."""
    from .views import predict

    df = pd.read_csv(csv)
    df["path"] = [str(images / i) for i in df.image_id]
    df = df[[Path(p).exists() for p in df.path]].copy()
    df["p_frontal"] = df.path.map(predict(list(df.path)))
    df = df[df.p_frontal >= 0.5]
    df["study"] = df.image_id.map(study_of)
    df = df.sort_values("p_frontal", ascending=False).drop_duplicates("study").sort_values("image_id")
    held = app_eval_studies(eval_dirs)
    df["split"] = [("test" if s in held else split_of(s)) for s in df.study]
    labels = pd.DataFrame([label(c) for c in df.org_caption.fillna("")], index=df.index)
    return pd.concat([df[["image_id", "path", "study", "split", "org_caption"]], labels], axis=1)


# --- the app's preprocessing -----------------------------------------------------------------


def enhance(gray: np.ndarray) -> np.ndarray:
    """Copy of the app's image_prep.enhance: stretch levels, then CLAHE."""
    lo, hi = np.percentile(gray, (1, 99))
    if hi - lo > 5:
        gray = np.clip((gray.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def to_tensor(gray: np.ndarray) -> torch.Tensor:
    """What the app's xray_model feeds the network: 1x224x224, scaled to [-1024, 1024]."""
    img = xrv.datasets.normalize(gray.astype(np.float32), 255)
    return torch.from_numpy(cv2.resize(img, (SIZE, SIZE))[None, ...])


# --- phone-photo augmentation ----------------------------------------------------------------


def photo(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make a clean scan look like a phone photo of film on a screen or light box: background
    around the film, a tilted camera, uneven light, glare, blur, noise, screen moire, JPEG."""
    h, w = gray.shape
    img = gray.astype(np.float32)

    # Background around the film and a camera that is not square-on.
    pad = int(rng.uniform(0.0, 0.15) * max(h, w))
    bg = rng.uniform(0, 255)
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)
    H, W = img.shape
    j = 0.08 * min(H, W)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + rng.uniform(-j, j, src.shape).astype(np.float32)
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=bg)
    a = rng.uniform(-8, 8)
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0), (W, H), borderValue=bg)

    # Exposure: darker or lighter, less contrast, a gamma curve.
    img = img * rng.uniform(0.6, 1.1) + rng.uniform(-30, 30)
    img = (img - img.mean()) * rng.uniform(0.6, 1.0) + img.mean()
    img = 255 * (np.clip(img, 0, 255) / 255) ** rng.uniform(0.7, 1.4)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Uneven light: a gradient across the photo.
    g = rng.uniform(-40, 40) * ((xx / W - 0.5) * np.cos(a) + (yy / H - 0.5) * np.sin(a))
    img = img + g
    # Glare: a soft bright spot from a window or flash.
    if rng.random() < 0.5:
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(0.05, 0.25) * min(H, W)
        img = img + rng.uniform(40, 150) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    # Moire from photographing a screen.
    if rng.random() < 0.3:
        f, t = rng.uniform(0.15, 0.6), rng.uniform(0, np.pi)
        img = img + rng.uniform(3, 12) * np.sin(f * (xx * np.cos(t) + yy * np.sin(t)))

    # Focus and shake.
    if rng.random() < 0.6:
        k = int(rng.choice([3, 5, 7]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    # Sensor noise.
    img = img + rng.normal(0, rng.uniform(1, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # Phone camera and messaging apps compress.
    q = int(rng.uniform(30, 90))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def light(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small changes for clean scans: a little rotation, zoom and shift."""
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-5, 5), rng.uniform(0.95, 1.1))
    m[:, 2] += rng.uniform(-0.04, 0.04, 2) * (w, h)
    return cv2.warpAffine(gray, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


class XrayDataset(Dataset):
    """mode: "clean" (as stored), "train" (light changes, plus phone photos with p_photo), or
    "photo" (every image as a phone photo, with a fixed seed per image so runs compare)."""

    def __init__(self, table: pd.DataFrame, mode: str = "clean", p_photo: float = 0.5, seed: int = 0):
        self.t = table.reset_index(drop=True)
        self.y = torch.tensor(self.t[PATHOLOGIES].to_numpy(dtype=np.float32))
        self.mode, self.p_photo, self.seed = mode, p_photo, seed

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int):
        gray = cv2.imread(self.t.path[i], cv2.IMREAD_GRAYSCALE)
        if self.mode == "train":
            rng = np.random.default_rng()
            gray = photo(gray, rng) if rng.random() < self.p_photo else light(gray, rng)
        elif self.mode == "photo":
            gray = photo(gray, np.random.default_rng(self.seed * 100_003 + i))
        return to_tensor(enhance(gray)), self.y[i]


class CaseDataset(Dataset):
    """The app's eval cases (folders with scan.jpg and report.txt), e.g. real phone photos."""

    def __init__(self, root: Path):
        self.cases = sorted(p for p in root.iterdir() if (p / "scan.jpg").exists() and (p / "report.txt").exists())
        rows = [label((c / "report.txt").read_text()) for c in self.cases]
        self.y = torch.tensor(pd.DataFrame(rows)[PATHOLOGIES].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, i: int):
        gray = cv2.imread(str(self.cases[i] / "scan.jpg"), cv2.IMREAD_GRAYSCALE)
        return to_tensor(enhance(gray)), self.y[i]
