# ScanArth fine-tuning

Fine-tunes the chest X-ray model the ScanArth app uses for hints, TorchXRayVision's DenseNet-121
(`densenet121-res224-all`, 18 findings), so it holds up on **phone photos of X-rays**, which is what
ScanArth users upload. The app is in the `x-ray-saathi` repo; there the model runs in
`backend/app/pipeline/xray_model.py` when `XRAY_MODEL=torchxrayvision`.

## How it works

- **Data:** the IU X-ray (Open-I) dataset: images plus each study's radiologist caption
  (`cleaned_dataset.csv` with `image_id, projection, org_caption`). By default it is read from the
  app repo's `data/archive/archive/`.
- **Views:** the dataset's `projection` column is wrong for about 1 in 5 images, so
  `ft/views.py` decides frontal or lateral from the image with TorchXRayVision's view classifier.
  One frontal image is kept per study (3,664 studies).
- **Labels:** `ft/labels.py` reads each caption and labels the 18 findings: present, absent, or
  doubtful ("possible effusion"). Doubtful labels are left out of the loss. A finding the caption
  does not mention counts as absent.
- **Split:** by study, fixed by a hash: 76% train, 10% val, 14% test. Studies used by the app's
  eval sets (`data/eval`, `data/eval-iu`, `data/eval-iu20`) always go to test.
- **Same preprocessing as the app:** grayscale, levels stretch + CLAHE, squash to 224×224,
  TorchXRayVision scaling. `ft/data.py` copies the app's `enhance()`; keep them the same.
- **Phone-photo simulation:** half of the training images are turned into fake phone photos:
  background around the film, a tilted camera, uneven light, glare, moiré, blur, noise and JPEG.
  Look at samples with the snippet under "Checks" below.
- **Rare findings:** findings with fewer than 40 positives in train (Pneumothorax, Mass,
  Consolidation and others) are not trained on directly. For those, the model is held to the
  original model's answer on the clean image, so it does not forget them. The same hold teaches it
  to answer the same for a photo as for the clean scan.
- **Operating points:** the app treats a score ≥ 0.5 as a hint. TorchXRayVision scales scores so
  0.5 is each finding's operating point; training picks new operating points on the validation
  set (clean and simulated photos together) and saves them inside the weights (`op_threshs`).
  Findings with at least 10 positives in validation get the point that best separates present from
  absent. Rarer ones get the point that only 2% of scans without the finding reach: without
  that, Pneumothorax and Pneumonia scored just over 0.5 on almost every photo.

## Setup

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Runs on Apple silicon (MPS), CUDA or CPU. The first run downloads the base weights to
`~/.torchxrayvision`.

## Run

```bash
.venv/bin/python -m ft.prepare                  # data/labels.csv + label counts per split
.venv/bin/python -m ft.train --out runs/v1      # about 15 minutes on an M5
.venv/bin/python -m ft.evaluate runs/v1/best.pt --cases ~/Documents/xray-saathi/data/eval
.venv/bin/python -m ft.calibrate runs/v1/best.pt  # only after changing operating_points()
```

`ft.train --help` lists the settings (epochs, learning rates, `--p-photo`, `--min-positives`,
`--keep`). `--limit 400 --epochs 1` is a one-minute check that everything runs.

`ft.evaluate` compares the base model with any number of checkpoints on:

- `clean`: the test split as stored;
- `photo`: the same images as simulated phone photos (fixed seed, so runs compare);
- `cases:<folder>`: real cases, folders with `scan.jpg` + `report.txt`, such as the app's 20
  phone photos in `data/eval`.

It prints mean AUC, and the two numbers that matter in the app: **caught** (present findings
scoring ≥ 0.5) and **false hints** (absent findings scoring ≥ 0.5).

## Results

`runs/v1`, the default settings, trained 2026-11-16 (517 test images; the 20 real phone photos have
enough positives for Lung Opacity only). Caught and false hints use the app's cut-off of 0.5:

| Model | Test set | Mean AUC | Caught | False hints |
|---|---|---|---|---|
| Base | clean | 0.792 | 84% | 45% |
| Base | simulated photo | 0.658 | 73% | 48% |
| Base | 20 real photos | 0.906 | 83% | 53% |
| **v1** | clean | **0.831** | 62% | **10%** |
| **v1** | simulated photo | **0.788** | 57% | **12%** |
| **v1** | 20 real photos | **0.922** | 58% | **8%** |

- On simulated photos, AUC goes up for every finding that has enough test positives, including
  those held to the base model (Consolidation 0.72 → 0.81, Edema 0.86 → 0.95).
- v1 gives a quarter as many false hints, but catches fewer findings. Raising the base model's
  cut-off to get the same false hints catches far fewer (at 0.6: 40% caught, 14% false hints on
  photos), so v1 is better at any balance. To move v1's balance, change `operating_points()` in
  `ft/model.py` (for example, the lowest point that keeps 90% sensitivity) and run `ft.calibrate`.
- On the 20 real photos the base model gives 196 hints in total, v1 gives 36. The base model hints
  Fracture and Pneumothorax on almost every photo; v1 gives no hints on most normal scans.
- Nodule and Lung Lesion stay near chance (0.53 to 0.60), with the base model or v1: small
  nodules don't survive a 224×224 photo, and captions mention many old granulomas.
- **In the app** (its eval on the 20 real photos: Sonnet writes, Opus judges), v1 gives the same
  result as the base model: main finding caught in 29% of cases, 0.2 false alarms per case.
  Most misses are findings the explanation leaves out or puts on the wrong side, which hints
  this weak don't change. The gain so far is cleaner input, not better explanations.
- An evaluation right after training gave v1 0.782 (photo) and 0.953 (real photos); every run
  since gives 0.788 and 0.922, while the base model's numbers never moved. The cause is unknown.

## Using a checkpoint in the app

`best.pt` holds the full weights, including the new operating points. In the app's
`xray_model.py`, after the model is built:

```python
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])
```

Nothing else changes: same input, same 18 outputs, same 0.5 cut-off. The app does this when
`XRAY_MODEL=torchxrayvision` and `XRAY_MODEL_WEIGHTS` is the path to a `best.pt`. On a server,
the file still has to be copied into the image or a volume.

## Checks

Look at the phone-photo simulation:

```bash
.venv/bin/python -c "
import cv2, numpy as np, pandas as pd
from ft.data import photo, enhance
g = cv2.imread(pd.read_csv('data/labels.csv').path[0], 0)
cv2.imwrite('photo_samples.png', np.hstack([cv2.resize(enhance(photo(g, np.random.default_rng(s))), (224, 224)) for s in range(6)]))"
```

## Limits

- **The labels come from captions,** by rules. They miss some findings and mislabel some.
  Better labels (CheXpert's labeller, or a radiologist checking the test split) would make both
  training and the scores more trustworthy.
- **Few positives:** the test split has fewer than 10 examples for several findings, so their AUCs
  move a lot between runs. Compare mean AUC and the caught / false-hint rates, not single labels.
- **The base model may have seen these images:** TorchXRayVision's `all` weights were trained on
  several public sets, which may include IU X-ray. Its `clean` scores here can be optimistic. The
  `photo` and `cases` sets are the fairer comparison.
- **Simulated photos are not real photos.** The 20 real phone photos in the app's eval are the
  real test, but they are too few to score most findings. More real phone photos with reports
  would help the most.
