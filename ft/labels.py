"""Turn a radiologist's caption into labels for the 18 TorchXRayVision findings.

Rule-based, like the CheXpert labeller but much smaller: find a finding's words in a sentence, then
decide whether the sentence denies it ("no pneumothorax"), doubts it ("possible effusion") or
states it. Each label is 1 (present), 0 (absent or not mentioned) or NaN (doubtful: left out of the
loss, the way TorchXRayVision trains on CheXpert's uncertain labels).

IU X-ray reports describe the whole chest, so a finding that is not mentioned is taken as absent.
"""

import math
import re

# The model's outputs, in its order (densenet121-res224-all).
PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema", "Emphysema",
    "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass",
    "Hernia", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]  # fmt: skip

_CARDIO = (
    r"cardiomegaly|enlarged (heart|cardiac silhouette)|"
    r"(heart|cardiac silhouette|heart size|cardiac size)( is| are|,)?( \w+)? (enlarged|increased)"
)
PATTERNS = {
    "Atelectasis": r"atelecta",
    "Consolidation": r"consolidat",
    "Infiltration": r"infiltrat",
    "Pneumothorax": r"pneumothora",
    "Edema": r"edema",
    "Emphysema": r"emphysema|copd|chronic obstructive",
    "Fibrosis": r"fibros|fibrotic",
    "Effusion": r"effusion",
    "Pneumonia": r"pneumonia",
    "Pleural_Thickening": r"pleural thickening|thickened pleura|pleural[- ]parenchymal scarring",
    "Cardiomegaly": _CARDIO,
    "Nodule": r"nodul",
    "Mass": r"\bmass(es)?\b",
    "Hernia": r"hernia",
    "Lung Lesion": r"nodul|\bmass(es)?\b|lesion",
    "Fracture": r"fractur",
    # Parent labels, as in CheXpert: an opacity of any cause; a big heart or a wide mediastinum.
    "Lung Opacity": r"opacit|airspace disease|consolidat|infiltrat|atelecta|edema|pneumonia",
    "Enlarged Cardiomediastinum": _CARDIO + r"|mediastin\w*( contour| silhouette)? (is )?(widen|enlarg)|widened mediastinum",
}
_COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}

# A denial before the words, anywhere in the same clause: "no focal consolidation, effusion or
# pneumothorax" denies all three.
NEGATION = re.compile(
    r"\b(no|not|without|negative for|free of|clear of|resolution of|resolved|removed|"
    r"rather than|instead of|exclude[sd]?)\b"
)
# A denial after the words: "the effusion has resolved".
NEGATION_AFTER = re.compile(r"\b(resolved|cleared|removed|no longer|not (seen|visible|identified))\b")
UNCERTAIN = re.compile(
    r"\b(may|might|possibl[ey]|cannot be excluded|cannot exclude|not excluded|questionable|"
    r"question of|suspicious|suggest\w*|equivocal|versus|vs|concern\w*|could)\b"
)
# A new clause starts after these, so a denial does not carry over: "no pneumothorax, but a small
# effusion".
CLAUSE = re.compile(r"[.;:]|\bbut\b|\bhowever\b|\balthough\b|\bwith\b")


def _clauses(text: str) -> list[str]:
    text = re.sub(r"x{2,}", " ", text.lower())  # IU's anonymised words are XXXX
    return [c.strip() for c in CLAUSE.split(text) if c.strip()]


def _state(clause: str, m: re.Match) -> float | None:
    before, after = clause[: m.start()], clause[m.end() :]
    if NEGATION.search(before) or NEGATION_AFTER.search(after):
        return 0.0
    if UNCERTAIN.search(clause):
        return math.nan
    return 1.0


def label(caption: str) -> dict[str, float]:
    """Labels for one caption: 1 present, 0 absent, NaN doubtful. Present beats doubtful beats absent
    across clauses ("no new effusion. small left effusion persists" is present)."""
    out = dict.fromkeys(PATHOLOGIES, 0.0)
    for clause in _clauses(caption):
        for name, rx in _COMPILED.items():
            for m in rx.finditer(clause):
                s = _state(clause, m)
                if s == 1.0:
                    out[name] = 1.0
                elif s is not None and math.isnan(s) and out[name] == 0.0:
                    out[name] = math.nan
    return out
