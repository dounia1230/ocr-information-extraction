"""Conservative field-to-line alignment and a real character TF-IDF/LinearSVC model."""

import pickle
import re
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .dataset import validate_prepared, write_json
from .extraction import (
    EXCLUDED_TOTAL,
    TOTAL_LABEL,
    amount_in_line,
    clean_lines,
    normalize_date,
)
from .schemas import LABELS, Receipt


def normalized_text(text: str) -> str:
    return re.sub(r"[^\w]", "", text.casefold())


def align_lines(text: str, fields: dict) -> list[dict]:
    """Exact normalized spans only. Ambiguous or partial matches are excluded."""
    lines = clean_lines(text)
    keys = [normalized_text(line) for line in lines]
    candidates: dict[str, list[tuple[int, int]]] = {label: [] for label in LABELS[:-1]}
    suspicious = set()
    for field in ("company", "address"):
        target = normalized_text(fields[field] or "")
        if not target:
            continue
        for start in range(len(lines)):
            for end in range(start + 1, len(lines) + 1):
                joined = "".join(keys[start:end])
                if joined == target:
                    candidates[field.upper()].append((start, end))
                if len(joined) >= len(target):
                    break
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
            label, reason = None, "partial or repeated annotation match"
        else:
            label, reason = "OTHER", "no annotation match; manually audit OCR errors"
        records.append({"line": i, "text": line, "label": label, "reason": reason})
    return records


def _examples(prepared: dict, split: str) -> list[dict]:
    records = []
    for doc_id in prepared["splits"][split]:
        doc = prepared["documents"][doc_id]
        records.extend(
            {"document": doc_id, **row} for row in align_lines(doc["text"], doc["fields"])
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
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)),
            (
                "svm",
                LinearSVC(
                    C=c, class_weight="balanced", random_state=prepared["seed"], max_iter=10000
                ),
            ),
        ]
    )
    pipeline.fit([row["text"] for row in train], [row["label"] for row in train])
    predicted = pipeline.predict([row["text"] for row in validation])
    actual = [row["label"] for row in validation]
    provenance = {
        split: {key: prepared["documents"][key]["fingerprint"] for key in prepared["splits"][split]}
        for split in ("train", "validation", "test")
    }
    bundle = {
        "format_version": 1,
        "pipeline": pipeline,
        "sklearn_version": sklearn.__version__,
        "provenance": provenance,
        "ocr_config": prepared["ocr_config"],
        "seed": prepared["seed"],
    }
    report = {
        "seed": prepared["seed"],
        "C": c,
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
    return bundle


def extract_svm(text: str, bundle: dict) -> Receipt:
    lines = clean_lines(text)
    if not lines:
        return Receipt()
    pipeline = bundle["pipeline"]
    margins = pipeline.decision_function(lines)
    predicted = pipeline.classes_[np.argmax(margins, axis=1)]
    scores = np.max(margins, axis=1)
    selected: dict[str, list[tuple[int, str, float]]] = {label: [] for label in LABELS}
    for i, (line, label, score) in enumerate(zip(lines, predicted, scores)):
        # A negative one-vs-rest margin provides no positive evidence for any field.
        if score > 0:
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
    dates = {value for _, line, _ in selected["DATE"] if (value := normalize_date(line))}
    if len(dates) == 1:
        values["date"] = dates.pop()
    amounts = {
        value for _, line, _ in selected["TOTAL"] if (value := amount_in_line(line)) is not None
    }
    if len(amounts) == 1:
        values["total"] = amounts.pop()
    return Receipt(**values)
