"""Load local SROIE variants and freeze reproducible document-level splits."""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .extraction import normalize_amount, normalize_date
from .ocr import configure_tesseract, recognize
from .preprocessing import document_pages
from .schemas import FIELDS, Receipt


@dataclass(frozen=True)
class SROIEDocument:
    id: str
    image: Path
    fingerprint: str
    fields: Receipt
    reference_text: str | None
    official_split: str | None


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def _official_split(path: Path) -> str | None:
    for part in path.parts[:-1]:
        if part.lower() in {"test", "test_data", "testing"}:
            return "test"
        if part.lower() in {"train", "train_data", "training"}:
            return "train"
    return None


def _pair(image: Path, candidates: list[Path], root: Path) -> Path | None:
    if not candidates:
        return None
    split = _official_split(image.relative_to(root))
    same_split = [p for p in candidates if _official_split(p.relative_to(root)) == split]
    candidates = same_split or [
        p for p in candidates if _official_split(p.relative_to(root)) is None
    ]
    if not candidates:
        return None
    scores = [
        len(set(image.relative_to(root).parts[:-1]) & set(p.relative_to(root).parts[:-1]))
        for p in candidates
    ]
    best = [p for p, score in zip(candidates, scores) if score == max(scores)]
    if len(best) != 1:
        raise ValueError(f"Ambiguous annotations for {image}: {best}")
    return best[0]


def load_sroie(root: str | Path) -> list[SROIEDocument]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"SROIE directory does not exist: {root}")
    images = []
    entities: dict[str, list[Path]] = {}
    boxes: dict[str, list[Path]] = {}
    raw_fields = {}
    references = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.append(path)
        elif path.suffix.lower() in {".txt", ".json"}:
            text = path.read_text(encoding="utf-8-sig").strip()
            if text.startswith("{"):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid annotation JSON in {path}: {exc}") from exc
                value = {key.lower(): item for key, item in value.items()}
                if not set(FIELDS).issubset(value):
                    raise ValueError(f"Annotation {path} must contain {', '.join(FIELDS)}.")
                entities.setdefault(path.stem.lower(), []).append(path)
                raw_fields[path] = value
            elif path.suffix.lower() == ".txt" and text:
                lines = []
                for row in text.splitlines():
                    columns = row.split(",", 8)
                    try:
                        if len(columns) != 9:
                            raise ValueError("expected eight coordinates and a transcription")
                        [int(column.strip()) for column in columns[:8]]
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid SROIE transcription row in {path}: {row}"
                        ) from exc
                    lines.append(columns[8].strip())
                boxes.setdefault(path.stem.lower(), []).append(path)
                references[path] = "\n".join(lines)
    if not images:
        raise ValueError(f"No SROIE images found under {root}.")
    documents = []
    seen = {}
    for image in images:
        annotation = _pair(image, entities.get(image.stem.lower(), []), root)
        if annotation is None:
            raise ValueError(f"Missing field annotation for {image}. Use a labeled SROIE subset.")
        raw = raw_fields[annotation]
        fields = Receipt(
            company=raw["company"],
            address=raw["address"],
            date=normalize_date(str(raw["date"])) if raw["date"] else None,
            total=normalize_amount(raw["total"]),
        )
        for field in ("date", "total"):
            if raw[field] not in (None, "") and getattr(fields, field) is None:
                raise ValueError(f"Cannot normalize {field}={raw[field]!r} in {annotation}.")
        fingerprint = hashlib.sha256(image.read_bytes()).hexdigest()
        if fingerprint in seen:
            raise ValueError(
                f"Duplicate image content: {image} and {seen[fingerprint]}; remove duplicates before splitting."
            )
        seen[fingerprint] = image
        transcript = _pair(image, boxes.get(image.stem.lower(), []), root)
        documents.append(
            SROIEDocument(
                id=image.relative_to(root).as_posix(),
                image=image,
                fingerprint=fingerprint,
                fields=fields,
                reference_text=references.get(transcript),
                official_split=_official_split(image.relative_to(root)),
            )
        )
    return documents


def split_documents(documents: list[SROIEDocument], seed: int = 42) -> dict[str, list[str]]:
    if len(documents) < 3:
        raise ValueError("At least three labeled documents are required for train/validation/test.")
    rng = random.Random(seed)
    test = [doc.id for doc in documents if doc.official_split == "test"]
    test_ids = set(test)
    remaining = sorted(doc.id for doc in documents if doc.id not in test_ids)
    rng.shuffle(remaining)
    if not test:
        test_size = max(1, round(len(documents) * 0.15))
        test, remaining = remaining[:test_size], remaining[test_size:]
    if len(remaining) < 2:
        raise ValueError("Need at least two non-test documents for training and validation.")
    validation_size = max(1, round(len(remaining) * 0.15))
    return {
        "train": sorted(remaining[validation_size:]),
        "validation": sorted(remaining[:validation_size]),
        "test": sorted(test),
    }


def prepare_dataset(
    root: str | Path, output: str | Path, *, seed: int = 42, language: str = "eng", psm: int = 6
) -> dict:
    output = Path(output)
    if (output / "splits.json").exists() or (output / "prepared.json").exists():
        raise ValueError(
            "Preparation already exists. Choose a new output directory to preserve the frozen split."
        )
    documents = load_sroie(root)
    splits = split_documents(documents, seed)
    version = configure_tesseract(language)
    # Freeze document membership before OCR or line-label generation.
    write_json(
        output / "splits.json",
        {
            "seed": seed,
            "splits": splits,
            "fingerprints": {doc.id: doc.fingerprint for doc in documents},
        },
    )
    prepared = {
        "format_version": 1,
        "seed": seed,
        "splits": splits,
        "ocr_config": {"language": language, "psm": psm, "tesseract_version": version},
        "documents": {},
    }
    for doc in documents:
        started = perf_counter()
        image = next(document_pages(doc.image))
        lines = recognize(image, language=language, psm=psm)
        elapsed = perf_counter() - started
        prepared["documents"][doc.id] = {
            "fingerprint": doc.fingerprint,
            "fields": doc.fields.model_dump(),
            "reference_text": doc.reference_text,
            "text": "\n".join(line.text for line in lines),
            "lines": [line.model_dump() for line in lines],
            "ocr_elapsed_seconds": elapsed,
        }
    write_json(output / "prepared.json", prepared)
    return prepared


def validate_prepared(prepared: dict) -> None:
    if prepared.get("format_version") != 1:
        raise ValueError("Unsupported prepared dataset. Run prepare with this application.")
    membership = []
    for split in ("train", "validation", "test"):
        ids = prepared["splits"][split]
        if not ids:
            raise ValueError(f"The {split} split is empty.")
        membership.extend(ids)
    if len(membership) != len(set(membership)):
        raise ValueError("Document leakage: train/validation/test splits overlap.")
    if set(membership) != set(prepared["documents"]):
        raise ValueError("Prepared documents and split membership differ.")
    fingerprints = [prepared["documents"][key]["fingerprint"] for key in membership]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Duplicate document fingerprints across the prepared dataset.")
