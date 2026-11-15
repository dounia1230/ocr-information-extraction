"""Conservative field-to-line alignment and a real character TF-IDF/LinearSVC model."""

import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .dataset import validate_prepared, write_json
from .extraction import (
    EXCLUDED_TOTAL,
    TOTAL_LABEL,
    amount_in_line,
    clean_lines,
    normalize_amount,
    normalize_date,
)
from .schemas import LABELS, Receipt


def normalized_text(text: str) -> str:
    return re.sub(r"[^\w]", "", text.casefold())


def line_features(lines: list[str]) -> list[str]:
    """Represent header placement and text shape without depending on specific years or prices."""
    features = []
    for index, line in enumerate(lines):
        text = re.sub(r"\b(?:rm|myr|usd)\b|\$", " currency ", line.casefold())
        text = re.sub(r"\d+", "0", text)
        position = f"header{index}" if index < 5 else "body"
        features.append(f"__{position}__ {' '.join(text.split())}")
    return features


def _text_spans(keys: list[str], target: str) -> tuple[list[tuple[int, int]], set[int]]:
    """Find exact annotations and exclude noisy near-matches from the OTHER class."""
    exact = []
    suspicious = set()
    for start in range(len(keys)):
        if not keys[start]:
            continue
        joined = ""
        for end in range(start + 1, len(keys) + 1):
            joined += keys[end - 1]
            if not keys[end - 1]:
                continue
            if len(joined) > 1.25 * len(target):
                break
            if joined == target:
                exact.append((start, end))
            elif len(target) >= 6 and len(joined) >= 0.8 * len(target):
                matcher = SequenceMatcher(None, target, joined, autojunk=False)
                if matcher.quick_ratio() < 0.8:
                    continue
                similarity = matcher.ratio()
                if similarity >= 0.8:
                    suspicious.update(range(start, end))
    return exact, suspicious


def align_lines(text: str, fields: dict) -> list[dict]:
    """Keep exact normalized labels and exclude ambiguous or likely corrupted field lines."""
    lines = clean_lines(text)
    keys = [normalized_text(line) for line in lines]
    candidates: dict[str, list[tuple[int, int]]] = {label: [] for label in LABELS[:-1]}
    suspicious = set()
    for field in ("company", "address"):
        target = normalized_text(fields[field] or "")
        if not target:
            continue
        matches, near_matches = _text_spans(keys, target)
        suspicious.update(near_matches)
        candidates[field.upper()].extend(matches)
        # OCR fragments of an unmatched annotated field must not become false OTHER labels.
        for i, key in enumerate(keys):
            if len(key) >= 4 and (key in target or target in key):
                suspicious.add(i)
    for i, line in enumerate(lines):
        if fields["date"] is not None and normalize_date(line) == fields["date"]:
            candidates["DATE"].append((i, i + 1))
        amount = amount_in_line(line)
        if fields["total"] is not None and amount == fields["total"]:
            candidates["TOTAL"].append((i, i + 1))
    # A labeled total resolves equal amounts on item/cash lines; otherwise exclude duplicates.
    totals = candidates["TOTAL"]
    labeled = [
        span
        for span in totals
        if TOTAL_LABEL.search(lines[span[0]]) and not EXCLUDED_TOTAL.search(lines[span[0]])
    ]
    if len(labeled) == 1:
        suspicious.update(start for start, _ in totals)
        candidates["TOTAL"] = labeled
    assigned: dict[int, list[str]] = {}
    ambiguous = set()
    for label, spans in candidates.items():
        if len(spans) != 1:
            for start, end in spans:
                ambiguous.update(range(start, end))
            continue
        start, end = spans[0]
        for i in range(start, end):
            assigned.setdefault(i, []).append(label)
    records = []
    for i, line in enumerate(lines):
        labels = assigned.get(i, [])
        if i in ambiguous or len(labels) > 1:
            label, reason = None, "ambiguous field match"
        elif len(labels) == 1:
            label, reason = labels[0], "unique normalized annotation match"
        elif i in suspicious:
            label, reason = None, "partial, repeated or noisy annotation match"
        else:
            label, reason = "OTHER", "no annotation match; manually audit OCR errors"
        records.append({"line": i, "text": line, "label": label, "reason": reason})
    return records


def _examples(prepared: dict, split: str) -> list[dict]:
    records = []
    for doc_id in prepared["splits"][split]:
        doc = prepared["documents"][doc_id]
        features = line_features(clean_lines(doc["text"]))
        records.extend(
            {"document": doc_id, **row, "model_text": features[row["line"]]}
            for row in align_lines(doc["text"], doc["fields"])
        )
    return records


def train_model(prepared: dict, model_path: str | Path, *, c: float = 1.0) -> dict:
    validate_prepared(prepared)
    if c <= 0:
        raise ValueError("SVM C must be positive.")
    train_rows = _examples(prepared, "train")
    validation_rows = _examples(prepared, "validation")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        model_path.with_suffix(".alignment.json"),
        {
            "train": train_rows,
            "validation": validation_rows,
            "warning": "Inspect these weak labels manually; unmatched noisy field lines may be OTHER.",
        },
    )
    train = [row for row in train_rows if row["label"] is not None]
    validation = [row for row in validation_rows if row["label"] is not None]
    missing = set(LABELS) - {row["label"] for row in train}
    if missing:
        raise ValueError(
            f"Training alignment has no examples for {sorted(missing)}. Inspect the alignment JSON or add labeled documents."
        )
    if not validation:
        raise ValueError("No aligned validation lines. Inspect the alignment JSON.")
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text",
                            TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True),
                        ),
                        (
                            "position",
                            CountVectorizer(token_pattern=r"__header[0-4]__|__body__", binary=True),
                        ),
                    ]
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=c, class_weight="balanced", random_state=prepared["seed"], max_iter=10000
                ),
            ),
        ]
    )
    pipeline.fit([row["model_text"] for row in train], [row["label"] for row in train])
    predicted = pipeline.predict([row["model_text"] for row in validation])
    actual = [row["label"] for row in validation]
    provenance = {
        split: {key: prepared["documents"][key]["fingerprint"] for key in prepared["splits"][split]}
        for split in ("train", "validation", "test")
    }
    bundle = {
        "format_version": 1,
        "feature_version": 2,
        "alignment_version": 2,
        "pipeline": pipeline,
        "sklearn_version": sklearn.__version__,
        "provenance": provenance,
        "ocr_config": prepared["ocr_config"],
        "seed": prepared["seed"],
    }
    report = {
        "seed": prepared["seed"],
        "C": c,
        "feature_version": 2,
        "alignment_version": 2,
        "features": "Character TF-IDF of normalized numbers/currencies plus first-five-line position indicators.",
        "train_documents": len(prepared["splits"]["train"]),
        "validation_documents": len(prepared["splits"]["validation"]),
        "train_lines": len(train),
        "excluded_train_lines": len(train_rows) - len(train),
        "validation_line_classification": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": LABELS,
        "validation_confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "note": "Scores use automatically aligned validation lines, not independent manual line labels. Evaluate final fields separately.",
    }
    joblib.dump(bundle, model_path)
    write_json(model_path.with_suffix(".report.json"), report)
    return report


def load_model(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Model not found: {path}. Run train first.")
    # Joblib uses pickle: only load artifacts you created or explicitly trust.
    try:
        bundle = joblib.load(path)
    except (EOFError, pickle.UnpicklingError, ValueError, KeyError, IndexError, ImportError) as exc:
        raise ValueError(
            f"Cannot read trained model {path}. Retrain with this application."
        ) from exc
    if (
        not isinstance(bundle, dict)
        or bundle.get("format_version") != 1
        or not isinstance(bundle.get("pipeline"), Pipeline)
    ):
        raise ValueError("Unsupported model file. Train it with this application.")
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise ValueError(
            "Model scikit-learn version differs. Retrain with the current uv environment."
        )
    if bundle.get("feature_version", 1) not in {1, 2}:
        raise ValueError("Unsupported model feature version. Retrain with this application.")
    return bundle


def _select_value(candidates: list[tuple[float, str | float | None]]) -> str | float | None:
    """Choose the strongest valid candidate, leaving exact score ties unresolved."""
    scores: dict[str | float, float] = {}
    for score, value in candidates:
        if value is not None:
            scores[value] = max(score, scores.get(value, float("-inf")))
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    if not ranked or (len(ranked) > 1 and scores[ranked[0]] == scores[ranked[1]]):
        return None
    return ranked[0]


def extract_svm(text: str, bundle: dict) -> Receipt:
    lines = clean_lines(text)
    if not lines:
        return Receipt()
    pipeline = bundle["pipeline"]
    inputs = line_features(lines) if bundle.get("feature_version", 1) == 2 else lines
    margins = pipeline.decision_function(inputs)
    predicted = pipeline.classes_[np.argmax(margins, axis=1)]
    scores = np.max(margins, axis=1)
    selected: dict[str, list[tuple[int, str, float]]] = {label: [] for label in LABELS}
    for i, (line, label, score) in enumerate(zip(lines, predicted, scores)):
        # Multiclass prediction uses argmax even when every one-vs-rest score is negative.
        if np.count_nonzero(margins[i] == score) != 1:
            continue
        selected[label].append((i, line, float(score)))
    values = {}
    company = selected["COMPANY"]
    if company:
        values["company"] = max(company, key=lambda item: item[2])[1]
    address = selected["ADDRESS"]
    if address:
        # Keep the strongest contiguous address block rather than joining distant predictions.
        anchor = max(address, key=lambda item: item[2])[0]
        by_index = {item[0]: item[1] for item in address}
        start = end = anchor
        while start - 1 in by_index:
            start -= 1
        while end + 1 in by_index:
            end += 1
        values["address"] = " ".join(by_index[i] for i in range(start, end + 1))
    values["date"] = _select_value(
        [(score, normalize_date(line)) for _, line, score in selected["DATE"]]
    )
    amounts = []
    for index, line, score in selected["TOTAL"]:
        value = amount_in_line(line)
        # Sparse OCR may separate a TOTAL-classified label from its numeric value.
        if value is None and index + 1 < len(lines):
            value = normalize_amount(lines[index + 1])
        if value is not None:
            amounts.append((score, value))
    values["total"] = _select_value(amounts)
    return Receipt(**values)
