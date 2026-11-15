# OCR Information Extraction

Reads a receipt (image or PDF) and extracts 4 fields as JSON: **company, date, address, total**.

Runs fully on your computer (CPU only). No API, no LLM, no GPU, no paid service.

**How it works:** Image/PDF → Tesseract OCR → Regex or SVM → JSON

## Install

1. Install Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install Tesseract (separate from Python):
   - Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng`
   - Windows: install it with English data and add it to PATH
3. Install the project:

```sh
uv sync --locked
uv run ocr-extract --help
```

## Use it

```sh
uv run ocr-extract extract receipt.jpg --output result.json
```

Works with `.png`, `.jpg`, `.jpeg`, `.pdf`.

Optional preprocessing flags: `--grayscale`, `--denoise`, `--threshold`, `--deskew`

Example output:

```json
{
  "company": "Example Store",
  "date": "2026-10-12",
  "address": null,
  "total": 125.5
}
```

If a field can't be found reliably, it is `null`.

## Train your own model (optional)

Download the [SROIE dataset](https://www.kaggle.com/datasets/tatevikhakobyan/sroie-datasetv1) into `data/raw/`, then:

```sh
uv run ocr-extract prepare data/raw --output data/processed
uv run ocr-extract train data/processed/prepared.json --model models/receipt-svm.joblib
uv run ocr-extract extract unseen.png --method svm --model models/receipt-svm.joblib
```

## Evaluate

```sh
uv run ocr-extract evaluate data/processed/prepared.json --dataset data/raw --model models/receipt-svm.joblib --output outputs/evaluation.json
```

This compares Regex vs SVM and gives precision, recall, F1 and exact match per field.

## Demo app

```sh
uv run streamlit run app.py
```

## Run tests

```sh
uv run pytest -q
```

## Limits

- Regex rules are made for receipts and assume day-first dates (12/10 = 12 October).
- No benchmark results are included; run the commands on your own data.
- Only load `.joblib` models you trust.