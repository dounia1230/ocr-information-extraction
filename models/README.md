# Included receipt SVM

`receipt-svm.joblib` is the trained model used by the default Streamlit model path.
It is stored directly in Git; a normal clone includes it. No SROIE download or training
is needed for inference. Install native Tesseract with English data and run:

```sh
uv sync --locked
uv run streamlit run app.py
```

Select **svm**. For CLI inference:

```sh
uv run ocr-extract extract receipt.jpg --method svm --model models/receipt-svm.joblib
```

## Artifact and compatibility

- Size: 6,722,520 bytes.
- SHA-256: `d000f781382e556855bc18fdbb9f6a5e233fb6c606ca64811d5f8bb8b379a813`.
- Python: 3.11; scikit-learn: 1.9.1; NumPy: 2.4.6; joblib: 1.6.0.
- Feature version: 2; alignment version: 2.
- Use the committed `uv.lock`. Loading rejects an incompatible scikit-learn version.
- Joblib uses pickle: only load model files you trust.

## Training and evaluation

Training data: [SROIE](https://www.kaggle.com/datasets/tatevikhakobyan/sroie-datasetv1).
Dataset images, OCR caches and line-level alignment dumps are not distributed here.

- CPU Tesseract OCR, English, PSM 6, original images.
- Frozen document split, seed 42: 521 train / 92 validation / 344 test receipts.
- Fifteen duplicate image copies removed before splitting, preferring test copies.
- One training receipt excluded because its address annotation was missing.
- 13,314 labeled training lines; 2,158 uncertain lines excluded.
- Positive company/address labels require normalized exact annotation matches.
  Likely OCR-corrupted field spans are excluded from OTHER instead of assigned guessed labels.
- Character TF-IDF (2-5 grams), normalized number/currency features, line-position
  indicators and balanced LinearSVC; C=0.5 selected on validation.
- Uploaded demonstration receipts were not used for training.

Measured on the 344 held-out SROIE receipts with original-image OCR:

| Field | Precision | Recall | F1 | Normalized exact match |
| --- | --- | --- | --- | --- |
| company | 0.3745 | 0.2558 | 0.3040 | 0.2558 |
| date | 0.9206 | 0.7413 | 0.8213 | 0.7413 |
| address | 0.2667 | 0.2558 | 0.2611 | 0.2558 |
| total | 0.7879 | 0.6047 | 0.6842 | 0.6047 |

Mean field F1: **0.5176**. Mean field exact match: **0.4644**.
Corpus OCR CER: 0.2382; WER: 0.4727.

Text matching casefolds and collapses whitespace while retaining punctuation.
Dates are normalized to ISO and totals to numeric values. OCR references keep
annotation-file row order. Incorrect non-null predictions count as false positives
and, when a true value exists, false negatives. Exact match includes null/null.
The summary report is in `receipt-svm.report.json`.

## Limits

Company and address accuracy remains limited, particularly on layouts unlike SROIE.
Missing or incorrect fields are possible. This model does not guarantee correct
extraction from every receipt. Numeric dates use the documented day-first convention
when ambiguous. SVM scores are not calibrated probabilities.

Raw training data, experimental models, backups and generated outputs stay ignored.
The optional training commands in the main README can replace this model locally;
commit an intentionally retrained artifact and updated report together.
