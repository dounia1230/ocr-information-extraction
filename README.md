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

## Try the included trained model

`models/receipt-svm.joblib` is included in this repository. You do **not** need to
download SROIE, create a Kaggle account, prepare data, or train a model to try SVM.
After installing Tesseract with English language data:

```sh
git clone https://github.com/dounia1230/ocr-information-extraction.git
cd ocr-information-extraction
uv sync --locked
uv run streamlit run app.py
```

Open the displayed local URL, choose **svm**, upload a receipt and click
**Extract information**. Keep the default model path `models/receipt-svm.joblib`.
From the terminal:

```sh
uv run ocr-extract extract receipt.jpg --method svm --model models/receipt-svm.joblib --output result.json
```

Use `uv sync --locked` to install the matching model dependencies. Model details,
measured scores and limitations are in [models/README.md](models/README.md).

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

Download and extract the [SROIE dataset](https://www.kaggle.com/datasets/tatevikhakobyan/sroie-datasetv1).
Use the directory containing `train` and `test`; for example, `data/sroie/SROIE2019`:

```sh
uv run ocr-extract prepare data/sroie/SROIE2019 --output data/processed --seed 42 --deduplicate
uv run ocr-extract train data/processed/prepared.json --model models/receipt-svm.joblib --c 0.5
uv run ocr-extract extract unseen.png --method svm --model models/receipt-svm.joblib
```

Wait until `prepare` reports that it saved `data/processed/prepared.json` before running
`train`. Preparation runs Tesseract on every receipt and prints progress. If extraction
used another folder, replace `data/sroie/SROIE2019` with that path.

`--deduplicate` keeps one copy of identical images, preferring the supplied test copy
to prevent train/test leakage. It records excluded paths in `splits.json` and leaves
raw files untouched. Without the flag, duplicates are rejected. Unrelated bundled
model files are ignored; only Tesseract and TF-IDF/SVM are used.

This SROIE release has a missing `address` label in `train/img/X51005663280.jpg`.
For this copy, add `--exclude-document train/img/X51005663280.jpg` to `prepare`.
Explicit exclusions are recorded before splitting; incomplete labels otherwise fail
validation. Single-byte Windows-1252 transcriptions are decoded with a diagnostic.

Compact and English month-name dates are supported. Signed totals are preserved for
refunds and adjustment notes. Training saves `receipt-svm.alignment.json` for manual
label inspection and `receipt-svm.report.json` for validation scores.

Positive company/address line labels require an exact match after normalization.
Likely OCR-corrupted matches (character similarity at least 0.8) are excluded from
training instead of being mislabeled as OTHER. Approximate text matching only uses
known dataset annotations during preparation of training/validation labels; it is
never used to look up values for an uploaded receipt. Audit excluded examples manually.

The classifier learns character TF-IDF patterns and line-position indicators. Numbers
and common currency markers are normalized for model features; returned values always
come from the original OCR text. There is no merchant lookup or Regex fallback in SVM
mode. Date/total candidates are selected by the model's score, and a total label may
refer to an amount on the following OCR line. Tied candidates remain null. SVM scores
are not calibrated probabilities. The model records its feature version; earlier
models remain readable, but retraining is needed to learn the new features.

`C=0.5` was selected on validation receipts; choose parameters on validation only.
The current label-filtered model trades a slightly lower validation mean field F1
(0.553 versus 0.554) for higher mean field exact match (0.497 versus 0.486).
SROIE contains a limited range of receipt layouts and languages. A trained model does
not guarantee correct extraction from every receipt. Broader support requires diverse
annotated training receipts and evaluation on documents kept out of training. Your
uploaded examples are not automatically added to training or used as fixed rules.

## Evaluate

```sh
uv run ocr-extract evaluate data/processed/prepared.json --dataset data/sroie/SROIE2019 --model models/receipt-svm.joblib --output outputs/evaluation.json
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

- Regex rules are made for receipts. Ambiguous dates remain day-first (12/10 = 12 October); unambiguous month-first dates also work (03/18/2026 = 18 March 2026).
- The included model has measured SROIE scores in `models/README.md`; accuracy on other receipt formats is not guaranteed.
- Only load `.joblib` models you trust.
