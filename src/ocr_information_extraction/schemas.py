"""Validated output shared by the CLI, evaluation and demonstration."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FIELDS = ("company", "date", "address", "total")
LABELS = ["COMPANY", "DATE", "ADDRESS", "TOTAL", "OTHER"]


class Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    company: str | None = None
    date: str | None = None
    address: str | None = None
    total: float | None = Field(default=None, strict=True)

    @field_validator("company", "address")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str | None) -> str | None:
        return date.fromisoformat(value).isoformat() if value is not None else None


class OCRWord(BaseModel):
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, width, height in the OCR image


class OCRLine(BaseModel):
    text: str
    words: list[OCRWord] = Field(default_factory=list)


class PageResult(BaseModel):
    page: int
    fields: Receipt
    text: str
    method: Literal["regex", "svm"]
    lines: list[OCRLine]
    elapsed_seconds: float
    warnings: list[str] = Field(default_factory=list)


class DocumentResult(BaseModel):
    source: str
    language: str
    psm: int
    pdf_dpi: int
    preprocessing: dict[str, bool]
    tesseract_version: str
    elapsed_seconds: float
    pages: list[PageResult]
