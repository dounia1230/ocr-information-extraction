"""One Tesseract TSV pass per page, retaining words and reading-order lines."""

import os
import shutil
from functools import lru_cache
from pathlib import Path

import pytesseract
from PIL import Image

from .schemas import OCRLine, OCRWord


@lru_cache(maxsize=8)
def configure_tesseract(language: str = "eng") -> str:
    command = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not command and os.name == "nt":
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR"
        if (candidate / "tesseract.exe").is_file():
            command = str(candidate / "tesseract.exe")
    if not command:
        raise RuntimeError(
            "Tesseract was not found. Install it or set TESSERACT_CMD to its binary."
        )
    pytesseract.pytesseract.tesseract_cmd = command
    try:
        version = str(pytesseract.get_tesseract_version())
        installed = set(pytesseract.get_languages(config=""))
    except (OSError, pytesseract.TesseractNotFoundError, pytesseract.TesseractError) as exc:
        raise RuntimeError(f"Cannot start Tesseract at {command}: {exc}") from exc
    missing = set(language.split("+")) - installed
    if missing:
        raise RuntimeError(f"Missing Tesseract language data: {', '.join(sorted(missing))}.")
    return version


def recognize(image: Image.Image, *, language: str = "eng", psm: int = 6) -> list[OCRLine]:
    configure_tesseract(language)
    if psm not in range(3, 14):
        raise ValueError("PSM must be between 3 and 13 (try 3, 6 or 11).")
    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
            timeout=120,
        )
    except (RuntimeError, OSError, pytesseract.TesseractError) as exc:
        raise RuntimeError(f"Tesseract OCR failed: {exc}") from exc
    grouped: dict[tuple[int, int, int], list[OCRWord]] = {}
    for i, token in enumerate(data["text"]):
        token = token.strip()
        if not token:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        word = OCRWord(
            text=token,
            confidence=float(data["conf"][i]),
            bbox=tuple(int(data[k][i]) for k in ("left", "top", "width", "height")),
        )
        grouped.setdefault(key, []).append(word)
    return [
        OCRLine(text=" ".join(w.text for w in words), words=words) for words in grouped.values()
    ]
