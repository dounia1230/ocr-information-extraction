"""Load local SROIE variants and freeze reproducible document-level splits."""

import copy
import hashlib
import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter

from PIL import Image

from .extraction import normalize_amount, normalize_date
from .ocr import configure_tesseract, recognize
from .preprocessing import document_pages
from .schemas import FIELDS, Receipt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetDocument:
    id: str
    image: Path
    fingerprint: str
    fields: Receipt
    reference_text: str | None
    official_split: str | None
    excluded_duplicates: tuple[str, ...] = ()
    reference_regions: tuple[dict, ...] = ()
    reference_labels_explicit: bool = False
    source: str = "sroie"
    evaluated_fields: tuple[str, ...] = FIELDS
    text_match: str = "standard"


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


def load_sroie(
    root: str | Path,
    *,
    deduplicate: bool = False,
    exclude_documents: list[str] | None = None,
) -> list[DatasetDocument]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"SROIE directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    images = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    excluded = set(exclude_documents or [])
    missing = excluded - {path.relative_to(root).as_posix() for path in images}
    if missing:
        raise ValueError(f"Excluded document IDs do not exist: {sorted(missing)}")
    images = [path for path in images if path.relative_to(root).as_posix() not in excluded]
    image_stems = {path.stem.lower() for path in images}
    entities: dict[str, list[Path]] = {}
    boxes: dict[str, list[Path]] = {}
    raw_fields = {}
    references = {}
    regions = {}
    for path in files:
        # Bundled tokenizer/model metadata is not a receipt annotation.
        if path.stem.lower() in image_stems and path.suffix.lower() in {".txt", ".json"}:
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw.decode("cp1252")
                logger.warning("Reading Windows-1252 annotation: %s", path)
            text = text.strip()
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
                boxes_for_file = []
                for row in text.splitlines():
                    columns = row.split(",", 8)
                    try:
                        if len(columns) != 9:
                            raise ValueError("expected eight coordinates and a transcription")
                        coordinates = [int(column.strip()) for column in columns[:8]]
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid SROIE transcription row in {path}: {row}"
                        ) from exc
                    transcription = columns[8].strip()
                    if not transcription:
                        continue
                    lines.append(transcription)
                    left, top = min(coordinates[::2]), min(coordinates[1::2])
                    right, bottom = max(coordinates[::2]), max(coordinates[1::2])
                    boxes_for_file.append(
                        {"text": transcription, "bbox": [left, top, right - left, bottom - top]}
                    )
                boxes.setdefault(path.stem.lower(), []).append(path)
                references[path] = "\n".join(lines)
                regions[path] = tuple(boxes_for_file)
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
        if fingerprint in seen and not deduplicate:
            raise ValueError(
                f"Duplicate image content: {image} and {seen[fingerprint]}; remove duplicates before splitting."
            )
        seen[fingerprint] = image
        transcript = _pair(image, boxes.get(image.stem.lower(), []), root)
        documents.append(
            DatasetDocument(
                id=image.relative_to(root).as_posix(),
                image=image,
                fingerprint=fingerprint,
                fields=fields,
                reference_text=references.get(transcript),
                official_split=_official_split(image.relative_to(root)),
                reference_regions=regions.get(transcript, ()),
            )
        )
    if deduplicate:
        groups: dict[str, list[DatasetDocument]] = {}
        for doc in documents:
            groups.setdefault(doc.fingerprint, []).append(doc)
        documents = []
        for group in groups.values():
            group.sort(key=lambda doc: (doc.official_split != "test", doc.id))
            documents.append(
                replace(group[0], excluded_duplicates=tuple(doc.id for doc in group[1:]))
            )
        documents.sort(key=lambda doc: doc.id)
    return documents


def split_documents(documents: list[DatasetDocument], seed: int = 42) -> dict[str, list[str]]:
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
    root: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    language: str = "eng",
    psm: int = 6,
    deduplicate: bool = False,
    exclude_documents: list[str] | None = None,
    workers: int = 1,
) -> dict:
    documents = load_sroie(root, deduplicate=deduplicate, exclude_documents=exclude_documents)
    return prepare_documents(
        documents,
        output,
        seed=seed,
        language=language,
        psm=psm,
        workers=workers,
        exclude_documents=exclude_documents,
    )


def prepare_documents(
    documents: list[DatasetDocument],
    output: str | Path,
    *,
    seed: int = 42,
    language: str = "eng",
    psm: int = 6,
    workers: int = 1,
    exclude_documents: list[str] | None = None,
) -> dict:
    output = Path(output)
    if (output / "prepared.json").exists():
        raise ValueError(
            "Preparation already exists. Choose a new output directory to preserve the frozen split."
        )
    if workers < 1:
        raise ValueError("Workers must be positive.")
    splits = split_documents(documents, seed)
    version = configure_tesseract(language)
    config = {"language": language, "psm": psm, "tesseract_version": version}
    if workers > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    excluded = {
        doc.id: list(doc.excluded_duplicates) for doc in documents if doc.excluded_duplicates
    }
    # Freeze document membership before OCR or line-label generation.
    manifest = {
        "seed": seed,
        "splits": splits,
        "fingerprints": {doc.id: doc.fingerprint for doc in documents},
        "excluded_duplicates": excluded,
        "excluded_documents": sorted(set(exclude_documents or [])),
        "ocr_config": config,
    }
    if (output / "splits.json").exists() and read_json(output / "splits.json") != manifest:
        raise ValueError(
            "The interrupted preparation has different inputs or OCR settings. Use a new output directory."
        )
    write_json(output / "splits.json", manifest)
    prepared = {
        "format_version": 1,
        "seed": seed,
        "splits": splits,
        "ocr_config": config,
        "documents": {},
        "excluded_duplicates": excluded,
        "excluded_documents": sorted(set(exclude_documents or [])),
        "execution": {"workers": workers, "omp_thread_limit": os.environ.get("OMP_THREAD_LIMIT")},
    }
    logger.info(
        "Preparing %s documents; splits: %s; excluded duplicate copies: %s",
        len(documents),
        {key: len(ids) for key, ids in splits.items()},
        sum(len(ids) for ids in excluded.values()),
    )

    def process(doc: DatasetDocument) -> tuple[str, dict]:
        cache = output / "ocr" / f"{doc.fingerprint}.json"
        if cache.exists():
            sample = read_json(cache)
            if sample.get("ocr_config") != config:
                raise ValueError(f"Incompatible OCR cache: {cache}")
        else:
            started = perf_counter()
            image = next(document_pages(doc.image))
            lines = recognize(image, language=language, psm=psm)
            sample = {
                "ocr_config": config,
                "image_size": list(image.size),
                "text": "\n".join(line.text for line in lines),
                "lines": [line.model_dump() for line in lines],
                "ocr_elapsed_seconds": perf_counter() - started,
            }
            write_json(cache, sample)
        return doc.id, {
            **sample,
            "fingerprint": doc.fingerprint,
            "fields": doc.fields.model_dump(),
            "reference_text": doc.reference_text,
            "reference_regions": list(doc.reference_regions),
            "reference_labels_explicit": doc.reference_labels_explicit,
            "source": doc.source,
            "evaluated_fields": list(doc.evaluated_fields),
            "text_match": doc.text_match,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, (doc_id, sample) in enumerate(executor.map(process, documents), start=1):
            prepared["documents"][doc_id] = sample
            if index == 1 or index % 25 == 0 or index == len(documents):
                logger.info("OCR %s/%s: %s", index, len(documents), doc_id)
    write_json(output / "prepared.json", prepared)
    return prepared


def enrich_prepared(prepared: dict, root: str | Path) -> dict:
    """Add reference boxes and image size without another OCR pass or a new split."""
    validate_prepared(prepared)
    documents = load_sroie(
        root, deduplicate=True, exclude_documents=prepared.get("excluded_documents", [])
    )
    indexed = {doc.id: doc for doc in documents}
    result = copy.deepcopy(prepared)
    for doc_id, sample in result["documents"].items():
        source = indexed.get(doc_id)
        if source is None or source.fingerprint != sample["fingerprint"]:
            raise ValueError(f"Cached image is missing or changed: {doc_id}")
        if source.fields.model_dump() != sample["fields"]:
            raise ValueError(f"Field annotations changed since the split was frozen: {doc_id}")
        with Image.open(source.image) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError(
                    f"Reference coordinates require normalized image orientation: {doc_id}"
                )
            sample["image_size"] = list(image.size)
        sample.update(
            reference_regions=list(source.reference_regions),
            reference_labels_explicit=False,
            source="sroie",
            evaluated_fields=list(FIELDS),
            text_match="standard",
        )
    return result


def load_wildreceipt(root: str | Path) -> list[DatasetDocument]:
    root = Path(root).resolve()
    names = {}
    for line in (root / "class_list.txt").read_text(encoding="utf-8-sig").splitlines():
        key, value = line.split(maxsplit=1)
        names[int(key)] = value.strip()
    mapping = {
        "Store_name_value": "COMPANY",
        "Store_addr_value": "ADDRESS",
        "Date_value": "DATE",
        "Total_value": "TOTAL",
    }
    documents = []
    for split in ("train", "test"):
        for line in (root / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines():
            raw = json.loads(line)
            image = (root / raw["file_name"]).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                raise ValueError(f"Invalid WildReceipt image path: {raw['file_name']}")
            regions = []
            for annotation in raw["annotations"]:
                coordinates = annotation["box"]
                left, top = min(coordinates[::2]), min(coordinates[1::2])
                right, bottom = max(coordinates[::2]), max(coordinates[1::2])
                category = names[annotation["label"]]
                regions.append(
                    {
                        "text": annotation["text"],
                        "bbox": [left, top, right - left, bottom - top],
                        "label": mapping.get(category, None if category == "Ignore" else "OTHER"),
                    }
                )
            regions.sort(
                key=lambda region: (region["bbox"][1] + region["bbox"][3] / 2, region["bbox"][0])
            )
            values = {}
            evaluated = list(FIELDS)
            for field in FIELDS:
                parts = [
                    region["text"]
                    for region in regions
                    if region["label"] == field.upper() and region["text"].strip()
                ]
                if field in {"company", "address"}:
                    values[field] = " ".join(parts) or None
                    continue
                normalizer = normalize_date if field == "date" else normalize_amount
                parsed = {normalizer(part) for part in parts}
                parsed.discard(None)
                values[field] = parsed.pop() if len(parsed) == 1 else None
                # Keep a unique parseable annotation; mask conflicting or wholly unparseable values.
                if parts and values[field] is None:
                    evaluated.remove(field)
                if field == "date" and parts:
                    for part in parts:
                        match = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}\b", part)
                        if (
                            match
                            and all(1 <= int(group) <= 12 for group in match.groups())
                            and match[1] != match[2]
                            and field in evaluated
                        ):
                            evaluated.remove(field)
            documents.append(
                DatasetDocument(
                    id=f"{split}/{image.relative_to(root).as_posix()}",
                    image=image,
                    fingerprint=hashlib.sha256(image.read_bytes()).hexdigest(),
                    fields=Receipt(**values),
                    reference_text="\n".join(
                        region["text"] for region in regions if region["text"]
                    ),
                    official_split=split,
                    reference_regions=tuple(regions),
                    reference_labels_explicit=True,
                    source="wildreceipt",
                    evaluated_fields=tuple(evaluated),
                    text_match="compact",
                )
            )
    # Preserve official test copies if the archive contains identical images under multiple IDs.
    grouped = {}
    for doc in sorted(documents, key=lambda doc: (doc.official_split != "test", doc.id)):
        if doc.fingerprint not in grouped:
            grouped[doc.fingerprint] = doc
        else:
            previous = grouped[doc.fingerprint]
            grouped[doc.fingerprint] = replace(
                previous, excluded_duplicates=(*previous.excluded_duplicates, doc.id)
            )
    return sorted(grouped.values(), key=lambda doc: doc.id)


def merge_prepared(base: dict, additional: dict) -> dict:
    """Keep base split membership intact and exclude every overlapping additional image."""
    validate_prepared(base)
    validate_prepared(additional)
    for key in ("language", "psm", "tesseract_version"):
        if base["ocr_config"][key] != additional["ocr_config"][key]:
            raise ValueError(f"Cannot mix OCR configurations with different {key}.")
    result = copy.deepcopy(base)
    fingerprints = {doc["fingerprint"] for doc in result["documents"].values()}
    excluded = []
    for split in ("train", "validation", "test"):
        for doc_id in additional["splits"][split]:
            doc = additional["documents"][doc_id]
            key = f"wildreceipt/{doc_id}"
            if doc["fingerprint"] in fingerprints:
                excluded.append(key)
                continue
            if key in result["documents"]:
                raise ValueError(f"Duplicate document ID: {key}")
            result["documents"][key] = doc
            result["splits"][split].append(key)
            fingerprints.add(doc["fingerprint"])
    result["additional_overlap_exclusions"] = excluded
    validate_prepared(result)
    return result


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
