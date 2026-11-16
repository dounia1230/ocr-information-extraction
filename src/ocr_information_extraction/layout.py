"""Receipt geometry, reference-label transfer and spatial SVM inputs."""

import math
import re
from collections import Counter
from statistics import mean

import numpy as np

from .extraction import EXCLUDED_TOTAL, TOTAL_LABEL, amount_in_line, normalize_date
from .schemas import LABELS, OCRLine


def bounds(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return [left, top, right - left, bottom - top]


def overlap_fraction(word: list[float], region: list[float]) -> float:
    left = max(word[0], region[0])
    top = max(word[1], region[1])
    right = min(word[0] + word[2], region[0] + region[2])
    bottom = min(word[1] + word[3], region[1] + region[3])
    return max(0, right - left) * max(0, bottom - top) / max(1, word[2] * word[3])


def document_rows(document: dict) -> list[dict]:
    """Keep OCR tokens, and use their actual vertical order when geometry exists."""
    lines = document.get("lines") or [
        {"text": line} for line in document["text"].splitlines() if line.strip()
    ]
    rows = []
    for index, line in enumerate(lines):
        if isinstance(line, OCRLine):
            line = line.model_dump()
        if not line["text"].strip():
            continue
        words = line.get("words", [])
        box = bounds([word["bbox"] for word in words])
        rows.append({"text": line["text"], "words": words, "bbox": box, "source_line": index})
    if rows and all(row["bbox"][3] > 0 for row in rows):
        rows.sort(key=lambda row: (row["bbox"][1] + row["bbox"][3] / 2, row["bbox"][0]))
    return rows


def _reference_labels(regions: list[dict], fields: dict) -> list[str | None]:
    from .ml import align_lines

    order = sorted(
        range(len(regions)),
        key=lambda i: (regions[i]["bbox"][1] + regions[i]["bbox"][3] / 2, regions[i]["bbox"][0]),
    )
    aligned = align_lines("\n".join(regions[i]["text"] for i in order), fields)
    labels = [None] * len(regions)
    for index, record in zip(order, aligned):
        labels[index] = record["label"]
    for i, region in enumerate(regions):
        if fields["date"] is not None and normalize_date(region["text"]) == fields["date"]:
            labels[i] = "DATE"
        if (
            fields["total"] is None
            or not TOTAL_LABEL.search(region["text"])
            or EXCLUDED_TOTAL.search(region["text"])
        ):
            continue
        if amount_in_line(region["text"]) == fields["total"]:
            labels[i] = "TOTAL"
            continue
        # SROIE often boxes TOTAL and its amount separately. Resolve repeated amounts
        # using the reference geometry rather than labeling every equal cash/item value.
        x, y, w, h = region["bbox"]
        candidates = []
        for j, other in enumerate(regions):
            bx, by, _, bh = other["bbox"]
            same_row = abs(by + bh / 2 - y - h / 2) <= 0.65 * max(h, bh)
            if (
                j != i
                and bx >= x + w
                and same_row
                and amount_in_line(other["text"]) == fields["total"]
            ):
                candidates.append(j)
        if len(candidates) == 1:
            labels[i] = labels[candidates[0]] = "TOTAL"
    return labels


def reference_line_labels(document: dict) -> list[dict]:
    """Transfer known reference-box labels to OCR words; never use OCR guesses as truth."""
    regions = document.get("reference_regions")
    if not regions:
        raise ValueError(
            "Reference boxes are missing. Enrich the cached dataset before training this stage."
        )
    if document.get("reference_labels_explicit"):
        labels = [region.get("label") for region in regions]
    else:
        labels = _reference_labels(regions, document["fields"])
    records = []
    for index, row in enumerate(document_rows(document)):
        counts = Counter()
        for word in row["words"]:
            matches = sorted(
                [
                    (overlap_fraction(word["bbox"], region["bbox"]), label)
                    for region, label in zip(regions, labels)
                ],
                reverse=True,
                key=lambda pair: pair[0],
            )
            if not matches or matches[0][0] < 0.5:
                continue
            best, label = matches[0]
            if any(score >= best - 0.1 and other != label for score, other in matches[1:]):
                continue
            counts[label] += 1
        positive = {label for label in counts if label not in {None, "OTHER"}}
        if len(positive) == 1:
            label = positive.pop()
            reason = "reference field region overlaps OCR words"
        elif len(positive) > 1:
            label, reason = None, "OCR line overlaps multiple annotated fields"
        elif counts["OTHER"] >= max(1, len(row["words"]) / 2):
            label, reason = "OTHER", "reference OTHER regions cover OCR words"
        else:
            label, reason = None, "insufficient or ambiguous reference coverage"
        records.append(
            {
                "line": index,
                "source_line": row["source_line"],
                "text": row["text"],
                "label": label,
                "reason": reason,
                "matched_words": sum(counts.values()),
                "word_count": len(row["words"]),
            }
        )
    return records


def _normalized(text: str) -> str:
    text = re.sub(r"\b(?:rm|myr|usd)\b|\$", " currency ", text.casefold())
    return re.sub(r"\d+", "0", text)


def spatial_inputs(document: dict) -> list[dict]:
    rows = document_rows(document)
    width, height = document.get("image_size", (0, 0))
    if rows and not (width > 0 and height > 0):
        # Older caches may have boxes but no image dimensions; keep their inputs bounded.
        width = max(row["bbox"][0] + row["bbox"][2] for row in rows)
        height = max(row["bbox"][1] + row["bbox"][3] for row in rows)
    width, height = max(1, width), max(1, height)
    inputs = []
    for index, row in enumerate(rows):
        text = row["text"]
        x, y, w, h = row["bbox"]
        previous = rows[index - 1] if index else None
        following = rows[index + 1] if index + 1 < len(rows) else None
        letters = [character for character in text if character.isalpha()]
        feature = {
            "text": _normalized(text),
            "previous_text": _normalized(previous["text"]) if previous else "__start__",
            "next_text": _normalized(following["text"]) if following else "__end__",
            "x": x / width,
            "y": y / height,
            "width": w / width,
            "height": h / height,
            "right": (x + w) / width,
            "bottom": (y + h) / height,
            "line_position": index / max(1, len(rows) - 1),
            "geometry_present": float(h > 0 and document.get("image_size") is not None),
            "uppercase_fraction": sum(c.isupper() for c in letters) / max(1, len(letters)),
            "digit_fraction": sum(c.isdigit() for c in text) / max(1, len(text)),
            "letter_fraction": len(letters) / max(1, len(text)),
            "starts_digit": float(text[0].isdigit()),
            "length": min(1.0, math.log1p(len(text)) / 6),
            "words": min(1.0, len(text.split()) / 20),
            "confidence": mean(max(0, word["confidence"]) / 100 for word in row["words"])
            if row["words"]
            else 0.0,
            "previous_gap": (y - previous["bbox"][1] - previous["bbox"][3]) / height
            if previous
            else 0.0,
            "next_gap": (following["bbox"][1] - y - h) / height if following else 0.0,
            "left_change": (x - previous["bbox"][0]) / width if previous else 0.0,
        }
        inputs.append(feature)
    return inputs


def text_columns(inputs: list[dict]) -> list[str]:
    return [row["text"] for row in inputs]


def layout_columns(inputs: list[dict]) -> list[dict]:
    # Neighbor text is kept in a separate sparse character vocabulary.
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"text", "previous_text", "next_text"}
        }
        for row in inputs
    ]


def context_columns(inputs: list[dict]) -> list[str]:
    return [f"previous={row['previous_text']} next={row['next_text']}" for row in inputs]


def transition_parameters(documents: dict, ids: list[str]) -> dict:
    counts = np.ones((len(LABELS), len(LABELS)))
    gaps = []
    for doc_id in ids:
        doc = documents[doc_id]
        rows = document_rows(doc)
        labels = reference_line_labels(doc)
        for i in range(1, len(labels)):
            left, right = labels[i - 1]["label"], labels[i]["label"]
            if left is not None and right is not None:
                counts[LABELS.index(left), LABELS.index(right)] += 1
            if left == right == "ADDRESS" and rows[i]["bbox"][3] > 0:
                gaps.append(
                    max(0, rows[i]["bbox"][1] - rows[i - 1]["bbox"][1] - rows[i - 1]["bbox"][3])
                    / max(1, rows[i - 1]["bbox"][3])
                )
    return {
        "log_transitions": np.log(counts / counts.sum(axis=1, keepdims=True)).tolist(),
        "address_gap_heights": float(np.quantile(gaps, 0.95)) if gaps else 1.5,
    }


def sequence_labels(
    scores: np.ndarray, classes: list[str], parameters: dict, weight: float
) -> list[str]:
    if not len(scores):
        return []
    order = [LABELS.index(label) for label in classes]
    transitions = np.asarray(parameters["log_transitions"])[np.ix_(order, order)]
    dp = scores[0].copy()
    back = []
    for emission in scores[1:]:
        options = dp[:, None] + weight * transitions
        previous = options.argmax(axis=0)
        dp = options.max(axis=0) + emission
        back.append(previous)
    state = int(dp.argmax())
    result = [state]
    for previous in reversed(back):
        state = int(previous[state])
        result.append(state)
    return [classes[i] for i in reversed(result)]
