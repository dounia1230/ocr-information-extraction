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


def field_metrics(
    truth: list[dict],
    predictions: list[dict],
    *,
    evaluated_fields: list[list[str]] | None = None,
    text_match: list[str] | None = None,
) -> dict:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Metrics require nonempty, equally sized truth and prediction lists.")
    evaluated_fields = evaluated_fields if evaluated_fields is not None else [FIELDS] * len(truth)
    text_match = text_match if text_match is not None else ["standard"] * len(truth)
    if len(evaluated_fields) != len(truth) or len(text_match) != len(truth):
        raise ValueError("Evaluation masks must match the number of documents.")
    results = {}
    for field in FIELDS:
        tp = fp = fn = exact = support = evaluated = 0
        for expected, predicted, mask, matching in zip(
            truth, predictions, evaluated_fields, text_match
        ):
            if field not in mask:
                continue
            evaluated += 1
            actual = normalize_field(field, expected[field])
            guess = normalize_field(field, predicted[field])
            if matching == "compact" and field in {"company", "address"}:
                actual = re.sub(r"\s+", "", actual) if actual is not None else None
                guess = re.sub(r"\s+", "", guess) if guess is not None else None
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
            "exact_match": exact / evaluated if evaluated else 0.0,
            "evaluated_documents": evaluated,
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


def check_model_splits(prepared: dict, model: dict, *, require_membership: bool = True) -> None:
    provenance = model["provenance"]
    all_hashes = [value for split in provenance.values() for value in split.values()]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("Model provenance contains duplicate documents across splits.")
    for split in ("train", "validation", "test"):
        for key in prepared["splits"][split]:
            fingerprint = prepared["documents"][key]["fingerprint"]
            if require_membership and provenance[split].get(key) != fingerprint:
                raise ValueError(
                    "Model and evaluation document splits differ. Use the frozen training dataset."
                )
    if model["ocr_config"] != prepared["ocr_config"]:
        raise ValueError("Model and prepared OCR configurations differ.")


def evaluate_cached(
    prepared: dict, model: dict | None = None, *, split: str = "validation"
) -> dict:
    """Compare extraction on fixed OCR, including disjoint external transfer tests."""
    validate_prepared(prepared)
    if split not in {"validation", "test"}:
        raise ValueError("Cached comparisons require validation or test documents.")
    ids = prepared["splits"][split]
    if model is not None:
        check_model_splits(prepared, model, require_membership=False)
        forbidden = set(model["provenance"]["train"].values())
        forbidden.update(model["provenance"]["validation" if split == "test" else "test"].values())
        if any(prepared["documents"][key]["fingerprint"] in forbidden for key in ids):
            raise ValueError(
                "Evaluation leakage: a scored document was used for training or tuning."
            )
    documents = [prepared["documents"][key] for key in ids]
    predictions, timings = [], []
    for doc in documents:
        started = perf_counter()
        receipt = (
            extract_regex(doc["text"])
            if model is None
            else extract_svm(doc["text"], model, document=doc)
        )
        timings.append(perf_counter() - started)
        predictions.append(receipt.model_dump())

    def summarize(indices):
        selected = [documents[i] for i in indices]
        fields = field_metrics(
            [doc["fields"] for doc in selected],
            [predictions[i] for i in indices],
            evaluated_fields=[doc.get("evaluated_fields", FIELDS) for doc in selected],
            text_match=[doc.get("text_match", "standard") for doc in selected],
        )
        return {
            "documents": len(selected),
            "fields": fields,
            "macro_f1": mean(score["f1"] for score in fields.values()),
            "mean_extraction_seconds": mean(timings[i] for i in indices),
            "mean_cached_ocr_seconds": mean(doc["ocr_elapsed_seconds"] for doc in selected),
        }

    result = summarize(list(range(len(ids))))
    result["sources"] = {
        source: summarize(
            [i for i, doc in enumerate(documents) if doc.get("source", "sroie") == source]
        )
        for source in sorted({doc.get("source", "sroie") for doc in documents})
    }
    result["predictions"] = [
        {
            "document": key,
            "expected": doc["fields"],
            "predicted": prediction,
            "evaluated_fields": doc.get("evaluated_fields", FIELDS),
        }
        for key, doc, prediction in zip(ids, documents, predictions)
    ]
    return result


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
        check_model_splits(prepared, model)
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
            "field_match": "Text: casefold and collapse whitespace, preserve punctuation. Dates: ISO, numeric day-first when ambiguous; month-first if day-first is invalid. Totals: numeric value.",
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
        texts, ocr_times, ocr_documents = [], [], []
        options = PreprocessOptions(**({variant: True} if variant != "original" else {}))
        for doc_id in test_ids:
            if variant == "original":
                texts.append(documents[doc_id]["text"])
                ocr_times.append(documents[doc_id]["ocr_elapsed_seconds"])
                ocr_documents.append(documents[doc_id])
            else:
                started = perf_counter()
                original = next(document_pages(root / doc_id))
                processed = preprocess(original, options)
                lines = recognize(processed, language=config["language"], psm=config["psm"])
                texts.append("\n".join(line.text for line in lines))
                ocr_times.append(perf_counter() - started)
                ocr_documents.append(
                    {
                        "text": texts[-1],
                        "lines": [line.model_dump() for line in lines],
                        "image_size": list(processed.size),
                    }
                )
        experiment = {
            "preprocessing": options.to_dict(),
            "ocr": ocr_metrics(references, texts),
            "methods": {},
        }
        for method in ["regex", "svm"] if model is not None else ["regex"]:
            predictions, timings, errors = [], [], []
            for index, text in enumerate(texts):
                started = perf_counter()
                fields = (
                    extract_regex(text)
                    if method == "regex"
                    else extract_svm(text, model, document=ocr_documents[index])
                )
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
