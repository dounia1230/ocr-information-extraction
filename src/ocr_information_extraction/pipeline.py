"""Complete image/PDF to validated JSON pipeline, independent of Streamlit."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from PIL import Image

from .extraction import extract_regex
from .ocr import configure_tesseract, recognize
from .preprocessing import PreprocessOptions, document_pages, preprocess
from .schemas import DocumentResult, PageResult


def process_document(
    source: str | Path | bytes,
    *,
    filename: str | None = None,
    options: PreprocessOptions | None = None,
    language: str = "eng",
    psm: int = 6,
    method: str = "regex",
    model=None,
    preview_dir: str | Path | None = None,
    on_page: Callable[[Image.Image, Image.Image], None] | None = None,
) -> DocumentResult:
    options = options or PreprocessOptions()
    if method not in {"regex", "svm"}:
        raise ValueError("Extraction method must be regex or svm.")
    if method == "svm" and model is None:
        raise ValueError("SVM extraction requires a trained model; run the train command first.")
    started = perf_counter()
    pages = []
    for number, original in enumerate(document_pages(source, filename=filename), start=1):
        page_started = perf_counter()
        processed = preprocess(original, options)
        if on_page is not None:
            on_page(original, processed)
        if preview_dir is not None:
            directory = Path(preview_dir)
            directory.mkdir(parents=True, exist_ok=True)
            original.save(directory / f"page-{number}-original.png")
            processed.save(directory / f"page-{number}-processed.png")
        lines = recognize(processed, language=language, psm=psm)
        text = "\n".join(line.text for line in lines)
        if method == "svm":
            from .ml import extract_svm

            fields = extract_svm(
                text,
                model,
                document={
                    "text": text,
                    "lines": [line.model_dump() for line in lines],
                    "image_size": list(processed.size),
                },
            )
        else:
            fields = extract_regex(text)
        warnings = []
        if not text:
            warnings.append("Tesseract recognized no text on this page.")
        elif method == "svm" and all(value is None for value in fields.model_dump().values()):
            warnings.append(
                "The SVM found no reliable fields in the recognized text. "
                "Unfamiliar layouts may need additional labeled training examples."
            )
        pages.append(
            PageResult(
                page=number,
                fields=fields,
                text=text,
                method=method,
                lines=lines,
                elapsed_seconds=perf_counter() - page_started,
                warnings=warnings,
            )
        )
    return DocumentResult(
        source=filename or str(source),
        language=language,
        psm=psm,
        pdf_dpi=150,
        preprocessing=options.to_dict(),
        tesseract_version=configure_tesseract(language),
        elapsed_seconds=perf_counter() - started,
        pages=pages,
    )
