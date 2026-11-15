# OCR Information Extraction

Reads a receipt (image or PDF) and extracts 4 fields as JSON: **company, date, address, total**.

Runs fully on your computer (CPU only). No API, no LLM, no GPU, no paid service.

**How it works:** Image/PDF → Tesseract OCR → Regex or SVM → JSON

## Install

1. Install Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Install Tesseract (separate from Python):
   - Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng`
   - Windows: install it with English data and add it to PATH
3. Clone and install the project:

```sh
git clone https://github.com/dounia1230/ocr-information-extraction.git
cd ocr-information-extraction
uv sync --locked
```

## Quick start

A trained model is already included (`models/receipt-svm.joblib`). No dataset or training needed.

**Demo app:**

```sh
uv run streamlit run app.py
```

Open the local URL, choose **svm**, upload a receipt, click **Extract information**.

**Command line:**

```sh
uv run ocr-extract extract receipt.jpg --method svm --model models/receipt-svm.joblib --output result.json
```

Without `--method svm`, it uses Regex rules instead.

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

Download and extract the [SROIE dataset](https://www.kaggle.com/datasets/tatevikhakobyan/sroie-datasetv1), then:

```sh
uv run ocr-extract prepare data/sroie/SROIE2019 --output data/processed --seed 42 --deduplicate --exclude-document train/img/X51005663280.jpg
uv run ocr-extract train data/processed/prepared.json --model models/receipt-svm.joblib --c 0.5
```

Wait for `prepare` to finish before running `train` (it runs OCR on every receipt).

Notes:
- `--deduplicate` removes identical images so train and test don't overlap.
- `--exclude-document` skips one receipt in this dataset that has a missing address label.

## Evaluate

```sh
uv run ocr-extract evaluate data/processed/prepared.json --dataset data/sroie/SROIE2019 --model models/receipt-svm.joblib --output outputs/evaluation.json
```

Compares Regex vs SVM: precision, recall, F1 and exact match per field.

## Run tests

```sh
uv run pytest -q
```

## Limits

- Dates like `12/10` are read as day-first (12 October). Unambiguous month-first dates (`03/18/2026`) also work.
- The model is trained on SROIE receipts only. It may fail on other layouts or languages.
- Scores are in [models/README.md](models/README.md).
- Only load `.joblib` models you trust.