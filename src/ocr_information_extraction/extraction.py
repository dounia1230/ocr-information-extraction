"""Small deterministic receipt rules and normalization shared with the SVM."""

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from math import isfinite

from .schemas import Receipt

DATE_PATTERN = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.](?:\d{4}|\d{2})"
    r"|\d{1,2}[- /][A-Za-z]{3,9}[- /](?:\d{4}|\d{2}))\b"
)
AMOUNT_PATTERN = re.compile(r"(?<![\w./-])\d+(?:[ .,]\d{3})*[.,]\d{2}(?!\d)")
TOTAL_LABEL = re.compile(
    r"\b(?:(?:grand|net)\s+total|total(?:\s+(?:due|payable))?|amount\s+due)\b", re.IGNORECASE
)
EXCLUDED_TOTAL = re.compile(
    r"\b(sub\s*total|subtotal|tax|gst|sst|saving|discount|change|cash|tendered)\b", re.IGNORECASE
)
ADDRESS_HINT = re.compile(
    r"\b(street|st\.?|road|rd\.?|jalan|jln\.?|lorong|taman|avenue|ave|lot|floor|"
    r"suite|block|no\.?|selangor|kuala\s+lumpur)\b",
    re.IGNORECASE,
)
NON_COMPANY = re.compile(
    r"\b(receipt|invoice|date|tel|phone|fax|cashier|welcome|thank|registration|gst|sst|reg)\b",
    re.IGNORECASE,
)


def clean_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def normalize_date(text: str) -> str | None:
    match = DATE_PATTERN.search(text)
    if not match:
        return None
    value = re.sub(r"[-/. ]", "-", match.group())
    # Numeric dates are day-first, as in SROIE; document this locale assumption.
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y", "%d-%b-%Y", "%d-%B-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_amount(value: str | float | None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    text = re.sub(r"^(?:RM|MYR|MAD|USD|EUR|\$)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace(" ", "")
    if not re.fullmatch(r"\d+(?:[.,]\d+)*", text):
        return None
    if "." in text and "," in text:
        decimal = "." if text.rfind(".") > text.rfind(",") else ","
        thousands = "," if decimal == "." else "."
        integer, fraction = text.rsplit(decimal, 1)
        if not re.fullmatch(rf"\d{{1,3}}(?:{re.escape(thousands)}\d{{3}})+", integer):
            return None
        text = integer.replace(thousands, "") + "." + fraction
    elif "," in text:
        if text.count(",") == 1 and len(text.rsplit(",", 1)[1]) <= 2:
            text = text.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
            text = text.replace(",", "")
        else:
            return None
    try:
        amount = Decimal(text)
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0 or amount.as_tuple().exponent < -2:
        return None
    result = float(amount)
    return result if isfinite(result) else None


def amount_in_line(line: str) -> float | None:
    matches = AMOUNT_PATTERN.findall(line)
    if matches:
        return normalize_amount(matches[-1])
    # Accept an integer only when the entire value or a total-labelled suffix is numeric.
    suffix = TOTAL_LABEL.split(line)[-1].strip(" :\t")
    return normalize_amount(suffix)


def extract_total(lines: list[str]) -> float | None:
    candidates = []
    for index, line in enumerate(lines):
        if not TOTAL_LABEL.search(line) or EXCLUDED_TOTAL.search(line):
            continue
        value = amount_in_line(line)
        if value is None and index + 1 < len(lines):
            value = normalize_amount(lines[index + 1])
        if value is not None:
            priority = 2 if re.search(r"grand|payable|due|net", line, re.IGNORECASE) else 1
            candidates.append((priority, value))
    if not candidates:
        return None
    highest_priority = max(item[0] for item in candidates)
    amounts = {value for priority, value in candidates if priority == highest_priority}
    return amounts.pop() if len(amounts) == 1 else None


def extract_regex(text: str) -> Receipt:
    lines = clean_lines(text)
    dates = {value for line in lines if (value := normalize_date(line))}
    company = None
    for line in lines[:5]:
        if (
            sum(c.isalpha() for c in line) >= 3
            and not NON_COMPANY.search(line)
            and not ADDRESS_HINT.search(line)
            and not TOTAL_LABEL.search(line)
            and not normalize_date(line)
            and not re.match(r"^\d", line)
        ):
            company = line
            break
    address = []
    for index, line in enumerate(lines):
        if ADDRESS_HINT.search(line) and not NON_COMPANY.search(line):
            address.append(line)
            for continuation in lines[index + 1 : index + 3]:
                if (
                    ADDRESS_HINT.search(continuation) or re.match(r"^\d{5}\b", continuation)
                ) and not NON_COMPANY.search(continuation):
                    address.append(continuation)
                else:
                    break
            break
    return Receipt(
        company=company,
        date=next(iter(dates)) if len(dates) == 1 else None,
        address=" ".join(address) or None,
        total=extract_total(lines),
    )
