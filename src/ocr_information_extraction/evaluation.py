"""Held-out field metrics and original/preprocessed OCR comparisons."""

import hashlib
import os
import platform
import re
from pathlib import Path
from statistics import mean
from time import perf_counter

import jiwer

from .dataset import validate_prepared
from .extraction import extract_regex, normalize_amount, normalize_date
from .ml import extract_svm
from .ocr import configure_tesseract, recognize
from .preprocessing import PreprocessOptions, document_pages, preprocess
from .schemas import FIELDS


def normalize_field(field: str, value):
    if value is None or value == "":
        return None
    if field == "total":
        return normalize_amount(value)
    if field == "date":
        return normalize_date(str(value))
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def field_metrics(truth: list[dict], predictions: list[dict]) -> dict:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Metrics require nonempty, equally sized truth and prediction lists.")
    results = {}
    for field in FIELDS:
        tp = fp = fn = exact = support = 0
        for expected, predicted in zip(truth, predictions):
            actual = normalize_field(field, expected[field])
            guess = normalize_field(field, predicted[field])
            same = actual == guess
            exact += same
            support += actual is not None
            tp += same and actual is not None
            fp += guess is not None and not same
            fn += actual is not None and not same
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        results[field] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "exact_match": exact / len(truth),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
    return results


def normalize_ocr(text: str) -> str:
    return " ".join(text.casefold().split())


def ocr_metrics(references: list[str | None], hypotheses: list[str]) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("OCR references and hypotheses must have equal lengths.")
    pairs = [
        (normalize_ocr(ref), normalize_ocr(hyp))
        for ref, hyp in zip(references, hypotheses)
        if ref is not None
    ]
    if not pairs:
        return {"cer": None, "wer": None, "reference_documents": 0}
    refs, hyps = zip(*pairs)
    return {
        "cer": jiwer.cer(list(refs), list(hyps)),
        "wer": jiwer.wer(list(refs), list(hyps)),
        "reference_documents": len(pairs),
    }


def evaluate_dataset(
    prepared: dict,
    root: str | Path,
    *,
    model: dict | None = None,
    variants: list[str] | None = None,
) -> dict:
    validate_prepared(prepared)
    variants = ["grayscale", "threshold"] if variants is None else variants
    if not set(variants).issubset({"grayscale", "threshold", "denoise", "deskew"}):
        raise ValueError("Unknown preprocessing experiment.")
    if model is not None:
        provenance = model["provenance"]
        for split in ("train", "validation", "test"):
            actual = {
                key: prepared["documents"][key]["fingerprint"] for key in prepared["splits"][split]
            }
            if provenance[split] != actual:
                raise ValueError(
                    "Model and evaluation document splits differ. Use the frozen training dataset."
                )
        if model["ocr_config"] != prepared["ocr_config"]:
            raise ValueError("Model and prepared OCR configurations differ.")
    root = Path(root).resolve()
    test_ids = prepared["splits"]["test"]
    documents = prepared["documents"]
    config = prepared["ocr_config"]
    version = configure_tesseract(config["language"])
    if version != config["tesseract_version"]:
        raise ValueError("Tesseract version changed since preparation. Prepare a fresh experiment.")
    for doc_id in test_ids:
        path = (root / doc_id).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Prepared document path leaves the dataset root.")
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != documents[doc_id]["fingerprint"]
        ):
            raise ValueError(f"Test image changed or is missing: {path}")
    report = {
        "seed": prepared["seed"],
        "split": "test",
        "documents": test_ids,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "ocr_config": config,
        "definitions": {
            "field_match": "Text: casefold and collapse whitespace, preserve punctuation. Dates: ISO, numeric day-first. Totals: numeric value.",
            "precision_recall": "TP=correct non-null field; wrong non-null adds FP, and FN if truth exists. Zero denominator gives 0.",
            "exact_match": "Fraction of documents matching per field, including null/null.",
            "ocr": "Corpus CER/WER; casefold and collapse whitespace; punctuation retained; transcription file row order.",
            "timing": "Wall-clock seconds for CPU-only loading/preprocessing/OCR plus extraction; original cached OCR timings reused. Not OS CPU user/system time.",
        },
        "experiments": {},
        "limitations": [
            "Automatically aligned line labels require manual inspection.",
            "A model trained on original OCR is evaluated unchanged on preprocessed OCR.",
            "Transcript box-file row order may differ from Tesseract reading order.",
        ],
    }
    truth = [documents[key]["fields"] for key in test_ids]
    references = [documents[key]["reference_text"] for key in test_ids]
    for variant in ["original", *dict.fromkeys(variants)]:
        texts, ocr_times = [], []
        options = PreprocessOptions(**({variant: True} if variant != "original" else {}))
        for doc_id in test_ids:
            if variant == "original":
                texts.append(documents[doc_id]["text"])
                ocr_times.append(documents[doc_id]["ocr_elapsed_seconds"])
            else:
                started = perf_counter()
                original = next(document_pages(root / doc_id))
                processed = preprocess(original, options)
                lines = recognize(processed, language=config["language"], psm=config["psm"])
                texts.append("\n".join(line.text for line in lines))
                ocr_times.append(perf_counter() - started)
        experiment = {
            "preprocessing": options.to_dict(),
            "ocr": ocr_metrics(references, texts),
            "methods": {},
        }
        for method in ["regex", "svm"] if model is not None else ["regex"]:
            predictions, timings, errors = [], [], []
            for index, text in enumerate(texts):
                started = perf_counter()
                fields = extract_regex(text) if method == "regex" else extract_svm(text, model)
                timings.append(ocr_times[index] + perf_counter() - started)
                prediction = fields.model_dump()
                predictions.append(prediction)
                failed = [
                    field
                    for field in FIELDS
                    if normalize_field(field, truth[index][field])
                    != normalize_field(field, prediction[field])
                ]
                if failed:
                    errors.append(
                        {
                            "document": test_ids[index],
                            "fields": failed,
                            "expected": truth[index],
                            "predicted": prediction,
                            "ocr_text": text,
                        }
                    )
            experiment["methods"][method] = {
                "fields": field_metrics(truth, predictions),
                "mean_processing_seconds": mean(timings),
                "failures": errors,
                "predictions": [
                    {"document": key, "fields": fields}
                    for key, fields in zip(test_ids, predictions)
                ],
            }
        report["experiments"][variant] = experiment
    return report


def comparison_markdown(report: dict) -> str:
    lines = [
        "| Image | Extractor | CER | WER | Mean field F1 | Mean field exact match | Seconds/doc |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, experiment in report["experiments"].items():
        cer = experiment["ocr"]["cer"]
        wer = experiment["ocr"]["wer"]
        cer_text = "N/A" if cer is None else f"{cer:.4f}"
        wer_text = "N/A" if wer is None else f"{wer:.4f}"
        for method, result in experiment["methods"].items():
            f1 = mean(field["f1"] for field in result["fields"].values())
            exact = mean(field["exact_match"] for field in result["fields"].values())
            lines.append(
                f"| {variant} | {method} | {cer_text} | {wer_text} | {f1:.4f} | {exact:.4f} | {result['mean_processing_seconds']:.3f} |"
            )
    return "\n".join(lines) + "\n"
