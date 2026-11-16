"""Geometry supervision, unseen inference, split isolation and public-data handling."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ocr_information_extraction import dataset
from ocr_information_extraction.dataset import DatasetDocument, load_wildreceipt, merge_prepared
from ocr_information_extraction.evaluation import evaluate_cached, field_metrics
from ocr_information_extraction.layout import document_rows, reference_line_labels, spatial_inputs
from ocr_information_extraction.ml import extract_svm, load_model, train_model
from ocr_information_extraction.schemas import FIELDS, LABELS, OCRLine, Receipt
from ocr_information_extraction.spatial import decode_document, grouped_folds


def receipt_document(texts=None):
    texts = texts or [
        "ALPHA MARKET",
        "12 MAIN STREET",
        "TOWN",
        "Date 13/10/2026",
        "TOTAL 12.50",
        "THANK YOU",
    ]
    labels = ["COMPANY", "ADDRESS", "ADDRESS", "DATE", "TOTAL", "OTHER"]
    lines, regions = [], []
    for i, (text, label) in enumerate(zip(texts, labels)):
        box = [10, 10 + 25 * i, 150, 15]
        lines.append({"text": text, "words": [{"text": text, "confidence": 90, "bbox": box}]})
        regions.append({"text": text, "bbox": box, "label": label})
    return {
        "text": "\n".join(texts),
        "lines": lines,
        "image_size": [200, 200],
        "reference_regions": regions,
        "reference_labels_explicit": True,
        "fields": {
            "company": "ALPHA MARKET",
            "address": "12 MAIN STREET TOWN",
            "date": "2026-10-13",
            "total": 12.5,
        },
        "ocr_elapsed_seconds": 0.1,
        "fingerprint": "placeholder",
    }


@pytest.fixture
def spatial_prepared():
    documents = {
        str(i): {**receipt_document(), "fingerprint": hashlib.sha256(str(i).encode()).hexdigest()}
        for i in range(12)
    }
    return {
        "format_version": 1,
        "seed": 42,
        "documents": documents,
        "splits": {
            "train": [str(i) for i in range(8)],
            "validation": ["8", "9"],
            "test": ["10", "11"],
        },
        "ocr_config": {"language": "eng", "psm": 6, "tesseract_version": "synthetic"},
    }


def test_reference_boxes_label_corrupted_ocr_and_exclude_mixed_fields():
    doc = receipt_document()
    doc["lines"][0]["text"] = "ALPNA MARKFT"
    assert reference_line_labels(doc)[0]["label"] == "COMPANY"
    doc["reference_regions"].append(
        {"text": "conflict", "bbox": [10, 10, 150, 15], "label": "ADDRESS"}
    )
    assert reference_line_labels(doc)[0]["label"] is None


def test_reference_total_resolves_equal_cash_value_in_separate_box():
    doc = receipt_document()
    doc["reference_labels_explicit"] = False
    doc["reference_regions"] = [
        {"text": "TOTAL", "bbox": [10, 110, 40, 15]},
        {"text": "12.50", "bbox": [100, 110, 60, 15]},
        {"text": "CASH", "bbox": [10, 135, 40, 15]},
        {"text": "12.50", "bbox": [100, 135, 60, 15]},
    ]
    doc["lines"] = [
        {
            "text": "TOTAL 12.50",
            "words": [
                {"text": "TOTAL", "confidence": 90, "bbox": [10, 110, 40, 15]},
                {"text": "12.50", "confidence": 90, "bbox": [100, 110, 60, 15]},
            ],
        }
    ]
    assert reference_line_labels(doc)[0]["label"] == "TOTAL"


def test_features_use_geometry_and_neighbors_without_annotations():
    doc = receipt_document()
    before = spatial_inputs(doc)
    doc.pop("reference_regions")
    doc.pop("fields")
    assert spatial_inputs(doc) == before
    assert before[1]["x"] == 0.05
    assert before[1]["previous_text"] == "alpha market"
    assert before[1]["next_text"] == "town"
    assert before[1]["text"] == "0 main street"


def test_older_ocr_cache_without_image_size_keeps_normalized_geometry_bounded():
    doc = receipt_document()
    doc.pop("image_size")
    features = spatial_inputs(doc)
    assert all(0 <= row["x"] <= row["right"] <= 1 for row in features)
    assert all(0 <= row["y"] <= row["bottom"] <= 1 for row in features)
    assert all(row["geometry_present"] == 0 for row in features)


def test_group_calibration_never_splits_a_receipt():
    labels = list(LABELS) * 6
    groups = [str(i) for i in range(6) for _ in LABELS]
    folds = grouped_folds(labels, groups)
    for train, held_out in folds:
        assert set(np.asarray(groups)[train]).isdisjoint(np.asarray(groups)[held_out])
    assert sorted(i for _, held_out in folds for i in held_out) == list(range(len(labels)))


@pytest.mark.parametrize("stage", ["references", "spatial", "blocks", "calibrated"])
def test_stages_reload_predict_unseen_values_without_teacher(spatial_prepared, tmp_path, stage):
    test_doc = spatial_prepared["documents"]["10"]
    test_doc["text"] += "\nzzzzzzzzzz"
    path = tmp_path / f"{stage}.joblib"
    report = train_model(spatial_prepared, path, c=0.5, stage=stage)
    bundle = load_model(path)
    doc = receipt_document(
        ["FRESH MARKET", "15 MAIN STREET", "TOWN", "Date 14/10/2035", "TOTAL 28.75", "THANK YOU"]
    )
    doc.pop("reference_regions")
    doc.pop("fields")
    result = extract_svm(doc["text"], bundle, document=doc)
    assert result.date == "2035-10-14"
    assert result.total == 28.75
    assert extract_svm("", bundle).company is None
    estimator = bundle["pipeline"]
    if stage == "calibrated":
        estimator = estimator.calibrated_classifiers_[0].estimator
        held_out = report["calibration"]["held_out_documents"]
        assert {key for fold in held_out for key in fold} == set(
            spatial_prepared["splits"]["train"]
        )
    vectorizer = dict(estimator.named_steps["features"].transformer_list)["text"]
    if stage != "references":
        vectorizer = vectorizer.named_steps["tfidf"]
    assert "zzzzz" not in vectorizer.vocabulary_


def test_blocks_merge_company_lines_stop_distant_address_and_apply_probability_cutoff():
    doc = receipt_document(["ALPHA", "MARKET", "12 MAIN STREET", "TOWN", "OTHER", "REMOTE"])
    rows = document_rows(doc)
    rows[3]["bbox"][1] = 500
    classes = np.array(sorted(LABELS))
    scores = np.full((6, 5), 0.01)
    for i, label in enumerate(["COMPANY", "COMPANY", "ADDRESS", "ADDRESS", "OTHER", "OTHER"]):
        scores[i, list(classes).index(label)] = 0.96 if i != 3 else 0.6
    bundle = {
        "pipeline": SimpleNamespace(classes_=classes),
        "stage": "blocks",
        "calibrated": True,
        "transitions": {"address_gap_heights": 2, "log_transitions": np.zeros((5, 5)).tolist()},
        "transition_weight": 0,
        "thresholds": {},
    }
    receipt, _ = decode_document(rows, scores, bundle)
    assert receipt.company == "ALPHA MARKET"
    assert receipt.address == "12 MAIN STREET"
    bundle["thresholds"] = {"company": 0.99}
    assert decode_document(rows, scores, bundle)[0].company is None


def test_masked_truth_is_not_counted_as_correct_null():
    truth = [{field: None for field in FIELDS}]
    predictions = [{**truth[0], "company": "ABC STORE"}]
    truth[0]["company"] = "ABCSTORE"
    score = field_metrics(
        truth, predictions, evaluated_fields=[["company"]], text_match=["compact"]
    )
    assert score["company"]["f1"] == 1
    assert score["date"]["evaluated_documents"] == 0
    assert score["date"]["exact_match"] == 0
    assert field_metrics(truth, predictions)["company"]["f1"] == 0


def test_repeated_company_uses_confidence_of_winning_block():
    classes = np.array(sorted(LABELS))
    doc = receipt_document(["ALPHA MARKET", "OTHER", "ALPHA MARKET", "OTHER", "OTHER", "OTHER"])
    scores = np.full((6, 5), 0.01)
    scores[:, list(classes).index("OTHER")] = 0.8
    company = list(classes).index("COMPANY")
    scores[0, company], scores[0, list(classes).index("OTHER")] = 0.6, 0.2
    scores[2, company], scores[2, list(classes).index("OTHER")] = 0.9, 0.05
    bundle = {
        "pipeline": SimpleNamespace(classes_=classes),
        "stage": "spatial",
        "calibrated": True,
        "thresholds": {"company": 0.8},
    }
    receipt, confidence = decode_document(document_rows(doc), scores, bundle)
    assert receipt.company == "ALPHA MARKET"
    assert confidence["company"] == 0.9


def test_merge_preserves_base_splits_and_excludes_overlapping_test_images(spatial_prepared):
    additional = copy.deepcopy(spatial_prepared)
    for key, doc in additional["documents"].items():
        doc["fingerprint"] = hashlib.sha256(f"additional-{key}".encode()).hexdigest()
    additional["documents"]["0"]["fingerprint"] = spatial_prepared["documents"]["10"]["fingerprint"]
    result = merge_prepared(spatial_prepared, additional)
    assert "wildreceipt/0" not in result["splits"]["train"]
    assert result["additional_overlap_exclusions"] == ["wildreceipt/0"]
    for split, ids in spatial_prepared["splits"].items():
        assert result["splits"][split][: len(ids)] == ids
    assert len(spatial_prepared["documents"]) == 12


def test_cached_comparison_rejects_training_and_tuning_images(spatial_prepared):
    provenance = {
        split: {key: spatial_prepared["documents"][key]["fingerprint"] for key in ids}
        for split, ids in spatial_prepared["splits"].items()
    }
    provenance["train"]["leaked"] = provenance["test"].pop("10")
    model = {"provenance": provenance, "ocr_config": spatial_prepared["ocr_config"]}
    with pytest.raises(ValueError, match="leakage"):
        evaluate_cached(spatial_prepared, model, split="test")


def test_wildreceipt_value_mapping_and_unknown_date_mask(tmp_path):
    (tmp_path / "class_list.txt").write_text(
        "0 Ignore\n1 Store_name_value\n3 Store_addr_value\n7 Date_value\n23 Total_value\n25 Others\n"
    )
    for split, color in (("train", "white"), ("test", "black")):
        Image.new("RGB", (200, 200), color).save(tmp_path / f"{split}.png")
        annotations = [
            {
                "box": [10, 10 + i * 25, 150, 10 + i * 25, 150, 25 + i * 25, 10, 25 + i * 25],
                "text": text,
                "label": label,
            }
            for i, (text, label) in enumerate(
                [("ABCSTORE", 1), ("12MAINSTREET", 3), ("01/09/2025", 7), ("$70", 23)]
            )
        ]
        record = {"file_name": f"{split}.png", "annotations": annotations}
        (tmp_path / f"{split}.txt").write_text(json.dumps(record) + "\n")
    docs = load_wildreceipt(tmp_path)
    assert len(docs) == 2
    assert docs[0].fields.total == 70
    assert docs[0].fields.company == "ABCSTORE"
    assert "date" not in docs[0].evaluated_fields
    assert docs[0].text_match == "compact"
    assert docs[0].reference_labels_explicit
    path = tmp_path / "train.txt"
    record["file_name"] = "../outside.png"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="outside|leaves|Invalid|missing"):
        load_wildreceipt(tmp_path)


def test_interrupted_preparation_resumes_ocr_without_changing_frozen_configuration(
    tmp_path, monkeypatch
):
    documents = []
    calls = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (20, 20), (index, 0, 0)).save(path)
        documents.append(
            DatasetDocument(
                id=path.name,
                image=path,
                fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(),
                fields=Receipt(),
                reference_text=None,
                official_split=None,
            )
        )
    monkeypatch.setattr(dataset, "configure_tesseract", lambda language: "synthetic")

    def recognize(image, **kwargs):
        index = image.getpixel((0, 0))[0]
        calls.append(index)
        if index == 1 and calls.count(1) == 1:
            raise RuntimeError("interrupted OCR")
        return [OCRLine(text="THANK YOU")]

    monkeypatch.setattr(dataset, "recognize", recognize)
    output = tmp_path / "processed"
    with pytest.raises(RuntimeError, match="interrupted"):
        dataset.prepare_documents(documents, output)
    with pytest.raises(ValueError, match="different inputs or OCR settings"):
        dataset.prepare_documents(documents, output, psm=3)
    result = dataset.prepare_documents(documents, output)
    assert len(result["documents"]) == 3
    assert calls.count(0) == calls.count(2) == 1
    assert calls.count(1) == 2
    with pytest.raises(ValueError, match="already exists"):
        dataset.prepare_documents(documents, output)
