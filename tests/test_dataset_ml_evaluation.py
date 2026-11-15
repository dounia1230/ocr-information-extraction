"""Synthetic fixtures validate mechanics; their scores are not SROIE results."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr_information_extraction import dataset, evaluation
from ocr_information_extraction.dataset import (
    load_sroie,
    prepare_dataset,
    read_json,
    split_documents,
    validate_prepared,
)
from ocr_information_extraction.evaluation import (
    comparison_markdown,
    evaluate_dataset,
    field_metrics,
    ocr_metrics,
)
from ocr_information_extraction.ml import align_lines, extract_svm, load_model, train_model
from ocr_information_extraction.schemas import OCRLine


@pytest.fixture
def local_sroie(tmp_path):
    root = tmp_path / "raw"
    for index in range(12):
        split = "train" if index < 10 else "test"
        for directory in ("img", "entities", "box"):
            (root / split / directory).mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (200, 120), "white")
        ImageDraw.Draw(image).text((10, 10), f"SYNTHETIC {index}", fill="black")
        image.save(root / split / "img" / f"receipt-{index}.jpg")
        fields = {
            "company": "ALPHA MARKET",
            "address": "12 MAIN STREET",
            "date": "12/10/2026",
            "total": "12.50",
        }
        (root / split / "entities" / f"receipt-{index}.txt").write_text(
            json.dumps(fields), encoding="utf-8"
        )
        (root / split / "box" / f"receipt-{index}.txt").write_text(
            "0,0,100,0,100,20,0,20,ALPHA MARKET\n"
            "0,20,100,20,100,40,0,40,12 MAIN STREET\n"
            "0,40,100,40,100,60,0,60,Date: 12/10/2026\n"
            "0,60,100,60,100,80,0,80,TOTAL 12.50\n"
            "0,80,100,80,100,100,0,100,THANK YOU, COME AGAIN",
            encoding="utf-8",
        )
    return root


@pytest.fixture
def prepared(tmp_path):
    documents = {}
    for index in range(12):
        path = tmp_path / f"synthetic-{index}.png"
        image = Image.new("RGB", (100, 100), (index * 10, 255, 255))
        image.save(path)
        fields = {
            "company": "ALPHA MARKET",
            "address": "12 MAIN STREET",
            "date": "2026-10-12",
            "total": 12.5,
        }
        text = "ALPHA MARKET\n12 MAIN STREET\nDate: 12/10/2026\nTOTAL 12.50\nTHANK YOU"
        documents[path.name] = {
            "fingerprint": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fields": fields,
            "text": text,
            "reference_text": text,
            "ocr_elapsed_seconds": 0.1,
        }
    keys = list(documents)
    return {
        "format_version": 1,
        "seed": 42,
        "splits": {"train": keys[:8], "validation": keys[8:10], "test": keys[10:]},
        "documents": documents,
        "ocr_config": {"language": "eng", "psm": 6, "tesseract_version": "test-version"},
    }


def test_sroie_loader_official_split_and_transcription(local_sroie):
    documents = load_sroie(local_sroie)
    assert len(documents) == 12
    assert all(doc.fields.date == "2026-10-12" for doc in documents)
    assert "THANK YOU, COME AGAIN" in documents[0].reference_text
    split = split_documents(documents)
    assert split == split_documents(documents)
    assert len(split["test"]) == 2
    assert all(doc_id.startswith("test/") for doc_id in split["test"])
    assert set(split["train"]).isdisjoint(split["validation"] + split["test"])


def test_duplicate_images_and_missing_annotations_fail(local_sroie):
    images = sorted(local_sroie.rglob("*.jpg"))
    original = images[1].read_bytes()
    images[1].write_bytes(images[0].read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        load_sroie(local_sroie)
    images[1].write_bytes(original)
    (local_sroie / "train/entities/receipt-0.txt").unlink()
    with pytest.raises(ValueError, match="Missing field annotation"):
        load_sroie(local_sroie)


def test_prepare_freezes_before_ocr(local_sroie, tmp_path, monkeypatch):
    output = tmp_path / "processed"
    monkeypatch.setattr(dataset, "configure_tesseract", lambda language: "test-version")

    def fake_ocr(image, **kwargs):
        assert (output / "splits.json").is_file()
        return [OCRLine(text="ALPHA MARKET"), OCRLine(text="TOTAL 12.50")]

    monkeypatch.setattr(dataset, "recognize", fake_ocr)
    prepared = prepare_dataset(local_sroie, output)
    validate_prepared(prepared)
    assert read_json(output / "prepared.json") == prepared
    with pytest.raises(ValueError, match="already exists"):
        prepare_dataset(local_sroie, output)


def test_loader_ignores_unrelated_model_metadata(local_sroie):
    bundled = local_sroie / "layoutlm-base-uncased"
    bundled.mkdir()
    (bundled / "config.json").write_text('{"model_type": "layoutlm"}', encoding="utf-8")
    (bundled / "vocab.txt").write_text("not a receipt transcription", encoding="utf-8")
    assert len(load_sroie(local_sroie)) == 12


def test_loader_reports_windows_1252_transcription(local_sroie, caplog):
    path = local_sroie / "train/box/receipt-0.txt"
    path.write_bytes("0,0,100,0,100,20,0,20,PRICE £12.50".encode("cp1252"))
    documents = load_sroie(local_sroie)
    doc = next(doc for doc in documents if doc.id == "train/img/receipt-0.jpg")
    assert doc.reference_text == "PRICE £12.50"
    assert "Windows-1252" in caplog.text


def test_explicit_exclusion_of_incomplete_annotation_is_audited(local_sroie, tmp_path, monkeypatch):
    annotation = local_sroie / "train/entities/receipt-0.txt"
    fields = read_json(annotation)
    del fields["address"]
    annotation.write_text(json.dumps(fields), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain"):
        load_sroie(local_sroie)
    excluded = ["train/img/receipt-0.jpg"]
    assert len(load_sroie(local_sroie, exclude_documents=excluded)) == 11
    monkeypatch.setattr(dataset, "configure_tesseract", lambda language: "test-version")
    monkeypatch.setattr(dataset, "recognize", lambda image, **kwargs: [OCRLine(text="TEST")])
    output = tmp_path / "excluded"
    result = prepare_dataset(local_sroie, output, exclude_documents=excluded)
    assert result["excluded_documents"] == excluded
    assert read_json(output / "splits.json")["excluded_documents"] == excluded
    with pytest.raises(ValueError, match="do not exist"):
        load_sroie(local_sroie, exclude_documents=["train/img/typo.jpg"])


def test_deduplication_preserves_test_and_audits_excluded_copies(
    local_sroie, tmp_path, monkeypatch
):
    test_image = local_sroie / "test/img/receipt-10.jpg"
    train_image = local_sroie / "train/img/receipt-0.jpg"
    train_image.write_bytes(test_image.read_bytes())
    original = train_image.read_bytes()
    documents = load_sroie(local_sroie, deduplicate=True)
    assert len(documents) == 11
    retained = next(doc for doc in documents if doc.id == "test/img/receipt-10.jpg")
    assert retained.excluded_duplicates == ("train/img/receipt-0.jpg",)
    assert train_image.read_bytes() == original
    monkeypatch.setattr(dataset, "configure_tesseract", lambda language: "test-version")
    monkeypatch.setattr(dataset, "recognize", lambda image, **kwargs: [OCRLine(text="TEST")])
    output = tmp_path / "deduplicated"
    result = prepare_dataset(local_sroie, output, deduplicate=True)
    assert result["excluded_duplicates"] == {"test/img/receipt-10.jpg": ["train/img/receipt-0.jpg"]}
    assert "train/img/receipt-0.jpg" not in result["documents"]
    assert read_json(output / "splits.json")["excluded_duplicates"] == result["excluded_duplicates"]


def test_annotation_is_never_borrowed_from_other_split(local_sroie):
    (local_sroie / "train/img/receipt-0.jpg").rename(local_sroie / "train/img/receipt-10.jpg")
    with pytest.raises(ValueError, match="Missing field annotation"):
        load_sroie(local_sroie)


def test_alignment_multiline_and_ambiguity():
    fields = {
        "company": "ABC STORE",
        "address": "12 MAIN STREET TOWN",
        "date": "2026-10-12",
        "total": 12.5,
    }
    rows = align_lines(
        "ABC STORE\n12 MAIN STREET\nTOWN\nDate 12/10/2026\nMilk 12.50\nTOTAL 12.50\nThanks", fields
    )
    assert [row["label"] for row in rows] == [
        "COMPANY",
        "ADDRESS",
        "ADDRESS",
        "DATE",
        None,
        "TOTAL",
        "OTHER",
    ]
    rows = align_lines("ABC STORE\nABC STORE\n12.50\n12.50", fields)
    assert all(row["label"] is None for row in rows)


def test_train_reload_unseen_inference_and_no_test_vocabulary(prepared, tmp_path):
    prepared["documents"][prepared["splits"]["test"][0]]["text"] += "\nzzzzzzzzzz"
    path = tmp_path / "model.joblib"
    report = train_model(prepared, path)
    bundle = load_model(path)
    assert report["train_lines"] == 40
    vectorizer = dict(bundle["pipeline"].named_steps["features"].transformer_list)["text"]
    assert "zzzzz" not in vectorizer.vocabulary_
    text = "FRESH MARKET\n15 MAIN STREET\nDate: 13/10/2026\nTOTAL 15.50\nTHANK YOU"
    prediction = extract_svm(text, bundle)
    assert prediction.total == 15.5
    assert prediction.date == "2026-10-13"
    assert extract_svm("", bundle).total is None
    assert path.with_suffix(".alignment.json").exists()
    assert report["validation_confusion_matrix"]
    with pytest.raises(ValueError, match="Model not found"):
        load_model(tmp_path / "missing.joblib")


def test_alignment_excludes_likely_ocr_fields_from_other():
    fields = {
        "company": "EXAMPLE MARKET",
        "address": "12 MAIN STREET TOWN",
        "date": None,
        "total": None,
    }
    rows = align_lines("EXAMPLE MARKFT\n12 MAIN STRFET\nTOWN\nTHANK YOU", fields)
    assert [row["label"] for row in rows] == [None, None, None, "OTHER"]
    assert rows[0]["reason"] == "partial, repeated or noisy annotation match"


def test_alignment_excludes_address_below_similarity_threshold():
    fields = {"company": None, "address": "12 MAIN STREET TOWN", "date": None, "total": None}
    rows = align_lines("12 MAIN STRFFT\nTOWN", fields)
    assert all(row["label"] is None for row in rows)


def test_alignment_excludes_repeated_approximate_matches():
    fields = {"company": "EXAMPLE MARKET", "address": None, "date": None, "total": None}
    rows = align_lines("EXAMPLE MARKFT\nOTHER TEXT\nEXAMPLE MARKFT", fields)
    assert rows[0]["label"] is None
    assert rows[2]["label"] is None
    assert rows[1]["label"] == "OTHER"


def test_exact_alignment_outranks_approximate_match_and_ignores_separator():
    fields = {"company": "EXAMPLE MARKET", "address": None, "date": None, "total": None}
    rows = align_lines("EXAMPLE MARKET\n---\nEXAMPLE MARKFT", fields)
    assert rows[0]["label"] == "COMPANY"
    assert rows[0]["reason"] == "unique normalized annotation match"
    assert rows[2]["label"] is None


def test_alignment_does_not_assign_positive_labels_to_unrelated_text():
    fields = {
        "company": "EXAMPLE MARKET",
        "address": "12 MAIN STREET TOWN",
        "date": None,
        "total": None,
    }
    rows = align_lines("UNRELATED SHOP\nSOMETHING ELSE\nTHANK YOU", fields)
    assert not any(row["label"] in {"COMPANY", "ADDRESS"} for row in rows)


def test_bad_splits_and_training_class_coverage(prepared, tmp_path):
    overlapping = copy.deepcopy(prepared)
    overlapping["splits"]["test"].append(overlapping["splits"]["train"][0])
    with pytest.raises(ValueError, match="leakage"):
        train_model(overlapping, tmp_path / "bad.joblib")
    for key in prepared["splits"]["train"]:
        prepared["documents"][key]["text"] = "THANK YOU"
    with pytest.raises(ValueError, match="no examples"):
        train_model(prepared, tmp_path / "missing-class.joblib")


def test_corrupt_model_is_actionable(tmp_path):
    path = tmp_path / "invalid.joblib"
    path.write_bytes(b"invalid model contents")
    with pytest.raises(ValueError, match="Cannot read trained model"):
        load_model(path)


@pytest.mark.parametrize("amount", ["$70", "RM 70.00", "-6.42"])
def test_svm_total_label_can_precede_value_with_negative_winning_margin(amount):
    classes = np.array(["ADDRESS", "COMPANY", "DATE", "OTHER", "TOTAL"])
    scores = np.array([[-2, -2, -2, -0.8, -0.2], [-2, -2, -2, 0.8, -1.0]])
    pipeline = SimpleNamespace(classes_=classes, decision_function=lambda lines: scores)
    fields = extract_svm(f"TOTAL:\n{amount}", {"pipeline": pipeline})
    assert fields.total == (-6.42 if amount.startswith("-") else 70)


def test_svm_total_does_not_take_unrelated_following_text():
    classes = np.array(["ADDRESS", "COMPANY", "DATE", "OTHER", "TOTAL"])
    scores = np.array([[-2, -2, -2, -0.8, -0.2], [-2, -2, -2, 0.8, -1.0]])
    pipeline = SimpleNamespace(classes_=classes, decision_function=lambda lines: scores)
    assert extract_svm("TOTAL:\nCARD # 1234", {"pipeline": pipeline}).total is None


def test_svm_selects_strongest_valid_date_and_leaves_tied_values_null():
    classes = np.array(["ADDRESS", "COMPANY", "DATE", "OTHER", "TOTAL"])
    scores = np.array([[-2, -2, 0.9, -1, -2], [-2, -2, 0.3, -1, -2]])
    pipeline = SimpleNamespace(classes_=classes, decision_function=lambda lines: scores)
    text = "Date 12/10/2026\nReturn before 19/10/2026"
    assert extract_svm(text, {"pipeline": pipeline}).date == "2026-10-12"
    scores[1, 2] = 0.9
    assert extract_svm(text, {"pipeline": pipeline}).date is None


def test_svm_does_not_choose_a_field_when_classes_tie():
    pipeline = SimpleNamespace(
        classes_=np.array(["ADDRESS", "COMPANY", "DATE", "OTHER", "TOTAL"]),
        decision_function=lambda lines: np.zeros((len(lines), 5)),
    )
    assert all(
        value is None
        for value in extract_svm("UNKNOWN TEXT", {"pipeline": pipeline}).model_dump().values()
    )


def test_svm_generalizes_numeric_values_and_currency_without_changing_output(prepared, tmp_path):
    for doc in prepared["documents"].values():
        doc["text"] = doc["text"].replace("TOTAL 12.50", "TOTAL RM 12.50")
    path = tmp_path / "normalized.joblib"
    train_model(prepared, path)
    bundle = load_model(path)
    fields = extract_svm(
        "FRESH MARKET\n15 MAIN STREET\nDate: 13/10/2035\nTOTAL $15.50\nTHANK YOU",
        bundle,
    )
    assert fields.date == "2035-10-13"
    assert fields.total == 15.5


def test_metrics_wrong_values_and_missing_references():
    truth = [
        {"company": "ABC  Store", "date": "2026-10-12", "address": None, "total": 10},
        {"company": None, "date": None, "address": None, "total": None},
    ]
    predictions = [
        {"company": "abc store", "date": "12/10/2026", "address": None, "total": 11},
        {"company": None, "date": None, "address": None, "total": 5},
    ]
    metrics = field_metrics(truth, predictions)
    assert metrics["company"]["precision"] == 1
    assert metrics["company"]["exact_match"] == 1
    assert metrics["total"]["false_positives"] == 2
    assert metrics["total"]["false_negatives"] == 1
    assert metrics["total"]["f1"] == 0
    assert ocr_metrics([None], ["some text"])["cer"] is None
    assert ocr_metrics(["ABC\nSTORE"], ["abc store"])["wer"] == 0
    assert ocr_metrics(["cat"], ["cut"])["cer"] == pytest.approx(1 / 3)


def test_evaluation_compares_same_test_and_rejects_model_leakage(prepared, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    train_model(prepared, path)
    bundle = load_model(path)
    monkeypatch.setattr(evaluation, "configure_tesseract", lambda language: "test-version")
    monkeypatch.setattr(
        evaluation, "recognize", lambda image, **kwargs: [OCRLine(text="TOTAL 12.50")]
    )
    report = evaluate_dataset(prepared, tmp_path, model=bundle)
    assert set(report["experiments"]) == {"original", "grayscale", "threshold"}
    assert report["documents"] == prepared["splits"]["test"]
    assert len(report["experiments"]["original"]["methods"]) == 2
    assert "| original | svm |" in comparison_markdown(report)
    bundle["provenance"]["train"] = {}
    with pytest.raises(ValueError, match="splits differ"):
        evaluate_dataset(prepared, tmp_path, model=bundle)
    (tmp_path / prepared["splits"]["test"][0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed or is missing"):
        evaluate_dataset(prepared, tmp_path)
