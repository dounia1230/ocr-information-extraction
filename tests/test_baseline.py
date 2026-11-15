from io import BytesIO

import numpy as np
import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from ocr_information_extraction import pipeline
from ocr_information_extraction.extraction import extract_regex, normalize_amount, normalize_date
from ocr_information_extraction.ocr import configure_tesseract
from ocr_information_extraction.pipeline import process_document
from ocr_information_extraction.preprocessing import (
    PreprocessOptions,
    document_pages,
    preprocess,
)
from ocr_information_extraction.schemas import DocumentResult, Receipt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Date: 12/10/2026", "2026-10-12"),
        ("2026.10.12", "2026-10-12"),
        ("12-Oct-26", "2026-10-12"),
        ("DATE: 03/18/2026 TIME: 10:52:41 PM", "2026-03-18"),
        ("03-18-26", "2026-03-18"),
        ("03/04/2026", "2026-04-03"),
        ("02/29/2024", "2024-02-29"),
        ("02/29/2025", None),
        ("20180304", "2018-03-04"),
        ("25032018", "2018-03-25"),
        ("Date: 20180428", "2018-04-28"),
        ("OCT 3, 2016", "2016-10-03"),
        ("Receipt no: 20180304", None),
        ("Phone: 25032018", None),
        ("31/02/2026", None),
        ("absent", None),
    ],
)
def test_dates(text, expected):
    assert normalize_date(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("RM 1,250.50", 1250.5),
        ("1.250,50", 1250.5),
        ("12,50", 12.5),
        ("1 250.50", 1250.5),
        ("0.00", 0),
        ("-12.00", -12),
        ("RM -1,250.50", -1250.5),
        ("--12.00", None),
        ("NaN", None),
        ("1,2,345", None),
        ("12,34.50", None),
        ("9" * 400, None),
        ("12.3456", None),
        (None, None),
    ],
)
def test_amounts(text, expected):
    assert normalize_amount(text) == expected


def test_receipt_rules_and_ambiguity():
    fields = extract_regex(
        "ABC STORE\n12 MAIN STREET\n12345 TOWN\nDate: 12/10/2026\n"
        "Subtotal 10.00\nTax 0.50\nTOTAL 10.50\nCash 20.00\nChange 9.50"
    )
    assert fields == Receipt(
        company="ABC STORE", date="2026-10-12", address="12 MAIN STREET 12345 TOWN", total=10.5
    )
    assert extract_regex("TOTAL 10.00\nTOTAL 12.00").total is None
    assert extract_regex("TOTAL\nRM 12,50").total == 12.5
    assert extract_regex("Subtotal 12.00\nCash 20.00").total is None
    assert extract_regex("Date 01/01/2026\nDate 02/01/2026").date is None
    assert extract_regex("") == Receipt()


def test_schema_rejects_invalid_fields():
    for values in (
        {"total": float("inf")},
        {"total": float("nan")},
        {"total": True},
        {"date": "yesterday"},
    ):
        with pytest.raises(ValidationError):
            Receipt(**values)
    assert Receipt(company="  ").company is None


@pytest.mark.parametrize("amount", ["-1.73", "- 1.73"])
def test_adjustment_receipt_preserves_negative_total(amount):
    fields = extract_regex(f"GARDENIA BAKERIES\nTotal 0% supplies: 12.98\nTotal Payable: {amount}")
    assert fields.total == -1.73
    assert Receipt.model_validate_json(fields.model_dump_json()).total == -1.73


@pytest.mark.parametrize("company", ["HEALTH PHARMACY", "PLAZA PHARMACY"])
def test_pharmacy_receipt_date_and_multiline_address(company):
    fields = extract_regex(
        f"{company}\n456 Medical Plaza\nHealthtown, USA 67890\n"
        "(555) 987-6543\nDATE: 03/18/2026 TIME: 10:52:41 PM\n"
        "RX# 12345678\nSUBTOTAL $21.98\nTOTAL $21.98\nPAYMENT VISA"
    )
    assert fields == Receipt(
        company=company,
        date="2026-03-18",
        address="456 Medical Plaza Healthtown, USA 67890",
        total=21.98,
    )


@pytest.mark.parametrize("metadata", ["RX# 12345", "PATIENT: PERSON 12345", "TEL: 12345"])
def test_address_does_not_capture_metadata_with_five_digit_numbers(metadata):
    fields = extract_regex(f"ABC STORE\n12 MAIN STREET\n{metadata}\nTOTAL 10.00")
    assert fields.address == "12 MAIN STREET"


def test_image_input_and_preprocessing_preserve_original():
    image = Image.new("RGB", (120, 80), "white")
    ImageDraw.Draw(image).text((10, 15), "TEST", fill="black")
    before = np.asarray(image).copy()
    for options in (
        PreprocessOptions(),
        PreprocessOptions(grayscale=True),
        PreprocessOptions(threshold=True, denoise=True, deskew=True),
    ):
        output = preprocess(image, options)
        assert output.size == image.size
        assert np.array_equal(np.asarray(image), before)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    assert next(document_pages(buffer.getvalue(), filename="receipt.PNG")).size == image.size
    with pytest.raises(ValueError, match="decode"):
        list(document_pages(b"invalid", filename="bad.png"))
    with pytest.raises(ValueError, match="Choose"):
        list(document_pages(b"invalid", filename="bad.txt"))


def test_pdf_pages_render_at_150_dpi():
    with pymupdf.open() as pdf:
        pdf.new_page(width=72, height=144)
        pdf.new_page(width=144, height=72)
        raw = pdf.tobytes()
    pages = list(document_pages(raw, filename="scan.pdf"))
    assert [page.size for page in pages] == [(150, 300), (300, 150)]
    with pytest.raises(ValueError, match="PDF"):
        list(document_pages(b"broken", filename="bad.pdf"))


def test_password_protected_pdf_fails_clearly():
    with pymupdf.open() as pdf:
        pdf.new_page()
        raw = pdf.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader"
        )
    with pytest.raises(ValueError, match="Password-protected"):
        list(document_pages(raw, filename="locked.pdf"))


def test_deskew_restores_horizontal_alignment():
    image = Image.new("RGB", (500, 250), "white")
    draw = ImageDraw.Draw(image)
    for top in (50, 100, 150, 200):
        draw.rectangle((40, top, 440, top + 4), fill="black")
    skewed = image.rotate(7, fillcolor="white")
    corrected = preprocess(skewed, PreprocessOptions(deskew=True))
    before = (np.asarray(skewed.convert("L")) < 128).sum(axis=1).max()
    after = (np.asarray(corrected) < 128).sum(axis=1).max()
    assert after > 2 * before


def test_svm_explains_when_recognized_text_yields_no_fields(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ocr_information_extraction.schemas import OCRLine

    path = tmp_path / "unknown.png"
    Image.new("RGB", (30, 30), "white").save(path)
    monkeypatch.setattr(pipeline, "recognize", lambda image, **kwargs: [OCRLine(text="UNKNOWN")])
    monkeypatch.setattr(pipeline, "configure_tesseract", lambda language: "test")
    classifier = SimpleNamespace(
        classes_=np.array(["ADDRESS", "COMPANY", "DATE", "OTHER", "TOTAL"]),
        decision_function=lambda lines: np.array([[-1, -1, -1, 1, -1]]),
    )
    result = process_document(path, method="svm", model={"pipeline": classifier})
    assert "no reliable fields" in result.pages[0].warnings[0]


@pytest.fixture
def synthetic_receipt():
    """Deliberately synthetic; never used as a SROIE benchmark."""
    image = Image.new("RGB", (1000, 620), "white")
    font = ImageFont.load_default(size=34)
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(
        [
            "ABC STORE",
            "12 MAIN STREET",
            "Date: 12/10/2026",
            "Milk 10.00",
            "TOTAL 10.50",
        ]
    ):
        draw.text((50, 45 + index * 95), line, font=font, fill="black")
    return image


@pytest.mark.integration
def test_real_tesseract_image_and_scanned_pdf(tmp_path, synthetic_receipt):
    try:
        configure_tesseract()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    image_path = tmp_path / "synthetic.png"
    synthetic_receipt.save(image_path)
    image_result = process_document(image_path, preview_dir=tmp_path / "preview")
    assert image_result.pages[0].fields.total == 10.5
    assert image_result.pages[0].fields.date == "2026-10-12"
    assert image_result.pages[0].lines[0].words[0].bbox[2] > 0
    assert DocumentResult.model_validate_json(image_result.model_dump_json()) == image_result
    with pymupdf.open() as pdf:
        for _ in range(2):
            page = pdf.new_page(width=480, height=300)
            page.insert_image(page.rect, filename=str(image_path))
        pdf_path = tmp_path / "synthetic.pdf"
        pdf.save(pdf_path)
    result = process_document(pdf_path, options=PreprocessOptions(threshold=True))
    assert [page.page for page in result.pages] == [1, 2]
    assert all(page.fields.total == 10.5 for page in result.pages)
