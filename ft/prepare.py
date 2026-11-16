"""Build data/labels.csv (one row per frontal image: split and 18 labels) and print label counts.

    python -m ft.prepare
"""

import argparse
from pathlib import Path

import numpy as np

from .data import load_table
from .labels import PATHOLOGIES

APP = Path.home() / "Documents/xray-saathi/data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=APP / "archive/archive/cleaned_dataset.csv")
    ap.add_argument("--images", type=Path, default=APP / "archive/archive/resized_images/512")
    ap.add_argument(
        "--eval-dirs", type=Path, nargs="*", default=[APP / "eval", APP / "eval-iu", APP / "eval-iu20"],
        help="the app's eval sets; their studies are kept out of train and val",
    )  # fmt: skip
    ap.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    a = ap.parse_args()

    t = load_table(a.csv, a.images, [d for d in a.eval_dirs if d.exists()])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)

    print(f"{len(t)} frontal images, {t.study.nunique()} studies -> {a.out}")
    print(t.split.value_counts().to_string(), "\n")
    print(f"{'label':28s}" + "".join(f"{s:>12s}" for s in ("train", "val", "test")) + "   (present / doubtful)")
    for p in PATHOLOGIES:
        cells = []
        for s in ("train", "val", "test"):
            v = t.loc[t.split == s, p].to_numpy()
            cells.append(f"{int((v == 1).sum()):>7d} /{int(np.isnan(v).sum()):>3d}")
        print(f"{p:28s}" + "".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()
