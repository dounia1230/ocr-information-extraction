"""Spatial TF-IDF/SVM experiments, block decoding and grouped probability calibration."""

from collections import Counter
from pathlib import Path
from statistics import mean

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from .dataset import validate_prepared, write_json
from .extraction import amount_in_line, normalize_amount, normalize_date
from .layout import (
    context_columns,
    document_rows,
    layout_columns,
    reference_line_labels,
    sequence_labels,
    spatial_inputs,
    text_columns,
    transition_parameters,
)
from .ml import _select_value
from .schemas import FIELDS, LABELS, Receipt


def _pipeline(c: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text",
                            Pipeline(
                                [
                                    ("select", FunctionTransformer(text_columns)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char", ngram_range=(2, 5), sublinear_tf=True
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "context",
                            Pipeline(
                                [
                                    ("select", FunctionTransformer(context_columns)),
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char",
                                            ngram_range=(2, 4),
                                            sublinear_tf=True,
                                            max_features=50000,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "layout",
                            Pipeline(
                                [
                                    ("select", FunctionTransformer(layout_columns)),
                                    ("vectorize", DictVectorizer()),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("svm", LinearSVC(C=c, class_weight="balanced", random_state=seed, max_iter=10000)),
        ]
    )


def _examples(prepared: dict, split: str) -> tuple[list[dict], list[dict]]:
    inputs, audit = [], []
    for doc_id in prepared["splits"][split]:
        doc = prepared["documents"][doc_id]
        features = spatial_inputs(doc)
        for row in reference_line_labels(doc):
            audit.append({"document": doc_id, **row})
            if row["label"] is not None:
                inputs.append(features[row["line"]])
    return inputs, audit


def grouped_folds(labels: list[str], groups: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit each calibrator on predictions from receipts absent from that fold's estimator."""
    if len(set(groups)) < 3:
        raise ValueError("Calibration needs at least three training documents.")
    folds = list(GroupKFold(n_splits=3).split(np.zeros(len(labels)), labels, groups))
    for train, held_out in folds:
        if set(np.asarray(labels)[train]) != set(LABELS):
            raise ValueError("A calibration fold lacks a field class; add more training documents.")
        if set(np.asarray(groups)[train]) & set(np.asarray(groups)[held_out]):
            raise ValueError("Calibration document groups overlap.")
    return folds


def score_document(document: dict, bundle: dict) -> tuple[list[dict], np.ndarray]:
    rows = document_rows(document)
    classes = list(bundle["pipeline"].classes_)
    if not rows:
        return rows, np.empty((0, len(classes)))
    inputs = spatial_inputs(document)
    pipeline = bundle["pipeline"]
    scores = (
        pipeline.predict_proba(inputs)
        if bundle.get("calibrated")
        else pipeline.decision_function(inputs)
    )
    return rows, np.asarray(scores)


def decode_document(rows: list[dict], scores: np.ndarray, bundle: dict) -> tuple[Receipt, dict]:
    """Select OCR values only. Reference text, boxes and field annotations are never read here."""
    if not rows:
        return Receipt(), {}
    classes = list(bundle["pipeline"].classes_)
    emissions = np.log(np.clip(scores, 1e-12, 1)) if bundle.get("calibrated") else scores
    if bundle.get("stage") in {"blocks", "calibrated"}:
        labels = sequence_labels(
            emissions, classes, bundle["transitions"], bundle.get("transition_weight", 0.0)
        )
    else:
        labels = [classes[index] for index in scores.argmax(axis=1)]
    selected = {label: [] for label in LABELS}
    for index, label in enumerate(labels):
        column = classes.index(label)
        if np.count_nonzero(scores[index] == scores[index, column]) > 1:
            continue
        selected[label].append(index)
    values, confidence = {}, {}
    blocks_enabled = bundle.get("stage") in {"blocks", "calibrated"}
    other = classes.index("OTHER")
    for field in ("company", "address"):
        label = field.upper()
        column = classes.index(label)
        blocks = []
        for index in selected[label]:
            join = bool(
                blocks and index == blocks[-1][-1] + 1 and (blocks_enabled or field == "address")
            )
            if join and blocks_enabled:
                previous = rows[blocks[-1][-1]]["bbox"]
                box = rows[index]["bbox"]
                gap = (box[1] - previous[1] - previous[3]) / max(1, previous[3])
                join = gap <= bundle["transitions"]["address_gap_heights"]
            if join:
                blocks[-1].append(index)
            else:
                blocks.append([index])
        candidates = []
        for block in blocks:
            strength = sum(float(emissions[i, column] - emissions[i, other]) for i in block)
            value = " ".join(rows[i]["text"] for i in block)
            candidates.append((strength, value))
        values[field] = _select_value(candidates)
        winning_score = max(
            (score for score, value in candidates if value == values[field]),
            default=float("-inf"),
        )
        chosen = next(
            (
                block
                for candidate, block in zip(candidates, blocks)
                if candidate == (winning_score, values[field])
            ),
            [],
        )
        confidence[field] = min(float(scores[i, column]) for i in chosen) if chosen else 0.0
    for field, parser in (("date", normalize_date), ("total", amount_in_line)):
        column = classes.index(field.upper())
        candidates = []
        for index in selected[field.upper()]:
            value = parser(rows[index]["text"])
            if field == "total" and value is None and index + 1 < len(rows):
                value = normalize_amount(rows[index + 1]["text"])
            candidates.append((float(scores[index, column]), value))
        values[field] = _select_value(candidates)
        confidence[field] = max(
            (score for score, value in candidates if value is not None and value == values[field]),
            default=0.0,
        )
    for field, threshold in bundle.get("thresholds", {}).items():
        if confidence.get(field, 0.0) < threshold:
            values[field] = None
    return Receipt(**values), confidence


def extract_document(document: dict, bundle: dict) -> Receipt:
    rows, scores = score_document(document, bundle)
    return decode_document(rows, scores, bundle)[0]


def _validation_fields(prepared: dict, cached: list, bundle: dict) -> dict:
    from .evaluation import field_metrics

    documents = [prepared["documents"][key] for key in prepared["splits"]["validation"]]
    predictions = [decode_document(rows, scores, bundle)[0].model_dump() for rows, scores in cached]
    return field_metrics(
        [doc["fields"] for doc in documents],
        predictions,
        evaluated_fields=[doc.get("evaluated_fields", FIELDS) for doc in documents],
        text_match=[doc.get("text_match", "standard") for doc in documents],
    )


def train_spatial(prepared: dict, model_path: str | Path, *, c: float, stage: str) -> dict:
    validate_prepared(prepared)
    if c <= 0:
        raise ValueError("SVM C must be positive.")
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    inputs, train_audit = _examples(prepared, "train")
    validation_inputs, validation_audit = _examples(prepared, "validation")
    train = [row for row in train_audit if row["label"] is not None]
    validation = [row for row in validation_audit if row["label"] is not None]
    labels = [row["label"] for row in train]
    if set(labels) != set(LABELS) or not validation:
        raise ValueError("Missing aligned classes or validation rows; inspect reference boxes.")
    write_json(
        model_path.with_suffix(".alignment.json"),
        {"train": train_audit, "validation": validation_audit},
    )
    estimator = _pipeline(c, prepared["seed"])
    folds = None
    if stage == "calibrated":
        groups = [row["document"] for row in train]
        folds = grouped_folds(labels, groups)
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=folds, ensemble=False)
    estimator.fit(inputs, labels)
    bundle = {
        "format_version": 1,
        "feature_version": 3,
        "alignment_version": 3,
        "stage": stage,
        "pipeline": estimator,
        "calibrated": stage == "calibrated",
        "sklearn_version": sklearn.__version__,
        "seed": prepared["seed"],
        "ocr_config": prepared["ocr_config"],
        "provenance": {
            split: {
                key: prepared["documents"][key]["fingerprint"] for key in prepared["splits"][split]
            }
            for split in ("train", "validation", "test")
        },
        "transitions": transition_parameters(prepared["documents"], prepared["splits"]["train"]),
        "transition_weight": 0.0,
        "thresholds": {},
    }
    # Cache predictions once: decoding/threshold experiments do not rerun OCR or vectorization.
    cached = [
        score_document(prepared["documents"][key], bundle)
        for key in prepared["splits"]["validation"]
    ]
    tuning = []
    weights = (0.0, 0.1, 0.25, 0.5) if stage in {"blocks", "calibrated"} else (0.0,)
    for weight in weights:
        bundle["transition_weight"] = weight
        fields = _validation_fields(prepared, cached, bundle)
        tuning.append(
            {"weight": weight, "macro_f1": mean(score["f1"] for score in fields.values())}
        )
    bundle["transition_weight"] = max(tuning, key=lambda row: row["macro_f1"])["weight"]
    if stage == "calibrated":
        for field in FIELDS:
            candidates = []
            for threshold in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8):
                bundle["thresholds"][field] = threshold
                score = _validation_fields(prepared, cached, bundle)[field]
                candidates.append((score["f1"], score["precision"], threshold))
            bundle["thresholds"][field] = max(
                candidates, key=lambda row: (row[0], row[1], -row[2])
            )[2]
    actual = [row["label"] for row in validation]
    predicted = estimator.predict(validation_inputs)
    report = {
        "seed": prepared["seed"],
        "C": c,
        "stage": stage,
        "feature_version": 3,
        "alignment_version": 3,
        "features": "Character TF-IDF of line and neighbors, normalized OCR geometry and text shape.",
        "train_documents": len(prepared["splits"]["train"]),
        "validation_documents": len(prepared["splits"]["validation"]),
        "train_lines": len(train),
        "excluded_train_lines": len(train_audit) - len(train),
        "train_class_counts": dict(Counter(labels)),
        "validation_line_classification": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": LABELS,
        "validation_confusion_matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
        "validation_fields": _validation_fields(prepared, cached, bundle),
        "transition_tuning": tuning,
        "transition_weight": bundle["transition_weight"],
        "thresholds": bundle["thresholds"],
        "note": "Line scores use transferred reference labels; final field scores use document annotations. All decoder choices use validation only.",
    }
    if folds is not None:
        probabilities = estimator.predict_proba(validation_inputs)
        classes = list(estimator.classes_)
        encoded = np.asarray([[float(label == name) for name in classes] for label in actual])
        report["calibration"] = {
            "method": "sigmoid; three document-group folds within training, ensemble=False",
            "validation_log_loss": log_loss(actual, probabilities, labels=classes),
            "validation_multiclass_brier": float(
                np.mean(np.sum((probabilities - encoded) ** 2, axis=1))
            ),
            "held_out_documents": [
                sorted({train[int(i)]["document"] for i in held_out}) for _, held_out in folds
            ],
        }
    joblib.dump(bundle, model_path)
    write_json(model_path.with_suffix(".report.json"), report)
    return report
