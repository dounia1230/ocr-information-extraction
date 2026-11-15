"""Image loading, page-wise PDF rendering and optional OpenCV experiments."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED = {".png", ".jpg", ".jpeg", ".pdf"}


@dataclass(frozen=True)
class PreprocessOptions:
    grayscale: bool = False
    denoise: bool = False
    threshold: bool = False
    deskew: bool = False

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def document_pages(
    source: str | Path | bytes, *, filename: str | None = None, dpi: int = 150
) -> Iterator[Image.Image]:
    """Yield independent RGB pages; keep a PDF open only during iteration."""
    name = filename if isinstance(source, bytes) else str(source)
    if not name or Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError("Choose a .png, .jpg, .jpeg or .pdf document.")
    if dpi <= 0:
        raise ValueError("PDF DPI must be positive.")
    if not isinstance(source, bytes) and not Path(source).is_file():
        raise ValueError(f"Document does not exist: {source}")
    if Path(name).suffix.lower() == ".pdf":
        try:
            document = (
                pymupdf.open(stream=source, filetype="pdf")
                if isinstance(source, bytes)
                else pymupdf.open(source)
            )
        except (pymupdf.FileDataError, RuntimeError) as exc:
            raise ValueError(f"Cannot open PDF: {exc}") from exc
        with document:
            if document.needs_pass:
                raise ValueError("Password-protected PDFs must be unlocked before processing.")
            if not len(document):
                raise ValueError("PDF has no pages.")
            for page in document:
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                yield Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return
    try:
        with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as image:
            image = ImageOps.exif_transpose(image)
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            yield Image.alpha_composite(background, rgba).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc


def preprocess(image: Image.Image, options: PreprocessOptions) -> Image.Image:
    """Never mutate the original. Filters run only when explicitly selected."""
    if not any(options.to_dict().values()):
        return image.copy()
    array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if options.denoise:
        array = cv2.fastNlMeansDenoising(array, h=10)
    if options.deskew:
        binary = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        points = cv2.findNonZero(binary)
        if points is not None and len(points) >= 5:
            angle = cv2.minAreaRect(points)[-1]
            if angle > 45:
                angle -= 90
            # Large rotations require orientation detection, outside this small-skew experiment.
            if 0.2 < abs(angle) <= 15:
                height, width = array.shape
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                array = cv2.warpAffine(
                    array, matrix, (width, height), flags=cv2.INTER_CUBIC, borderValue=255
                )
    if options.threshold:
        array = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(array)
