"""Run with: uv run streamlit run app.py"""

from pathlib import Path

import streamlit as st

from ocr_information_extraction.ml import load_model
from ocr_information_extraction.pipeline import process_document
from ocr_information_extraction.preprocessing import PreprocessOptions
from ocr_information_extraction.schemas import DocumentResult


@st.cache_resource
def cached_model(path: str, modified: int):
    return load_model(path)


st.set_page_config(page_title="Receipt OCR", page_icon="🧾", layout="wide")
st.title("Receipt OCR & information extraction")
st.caption("Sujet 4 · Local Tesseract on CPU · Regex and supervised TF-IDF + SVM")

with st.sidebar:
    st.header("Recognition")
    language = st.text_input("Tesseract language", "eng")
    psm = st.selectbox(
        "Page segmentation mode",
        [6, 3, 11],
        help="6: a text block · 3: automatic layout · 11: sparse text",
    )
    st.header("Preprocessing")
    st.caption("Keep all options off for the original-image baseline. Filters use grayscale.")
    options = PreprocessOptions(
        grayscale=st.checkbox("Grayscale"),
        denoise=st.checkbox("Denoise"),
        threshold=st.checkbox("Otsu threshold"),
        deskew=st.checkbox("Correct small skew"),
    )
    st.header("Extraction")
    model_path = st.text_input("Local trained model", "models/receipt-svm.joblib")
    model_file = Path(model_path)
    has_model = model_file.is_file()
    method = st.selectbox("Method", ["regex", "svm"] if has_model else ["regex"])
    if not has_model:
        st.info("SVM becomes available after training a model. See the README training command.")
    st.caption("Load only model files you created or trust.")

uploaded = st.file_uploader(
    "Upload a receipt image or scanned PDF", type=["png", "jpg", "jpeg", "pdf"]
)
st.caption("PDFs are rendered page by page at 150 DPI. Missing fields remain null.")
if uploaded is not None and st.button("Extract information", type="primary"):
    st.session_state.pop("analysis", None)
    try:
        model = (
            cached_model(str(model_file.resolve()), model_file.stat().st_mtime_ns)
            if method == "svm"
            else None
        )
        previews = []
        with st.spinner("Reading the document with Tesseract…"):
            result = process_document(
                uploaded.getvalue(),
                filename=uploaded.name,
                options=options,
                language=language,
                psm=psm,
                method=method,
                model=model,
                on_page=lambda original, processed: previews.append((original, processed)),
            )
        st.session_state["analysis"] = {"result": result.model_dump(), "previews": previews}
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        st.error(str(exc))

if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    result = DocumentResult.model_validate(analysis["result"])
    st.subheader(f"Results · {result.source}")
    st.caption(
        f"{result.language} · PSM {result.psm} · {result.elapsed_seconds:.2f} seconds · Settings recorded in JSON"
    )
    for page, (original, processed) in zip(result.pages, analysis["previews"]):
        with st.expander(f"Page {page.page} · {page.method}", expanded=True):
            left, right = st.columns(2)
            left.image(original, caption="Original", width="stretch")
            right.image(processed, caption="OCR input", width="stretch")
            st.table(
                [
                    {"Field": key, "Value": str(value) if value is not None else "null"}
                    for key, value in page.fields.model_dump().items()
                ]
            )
            st.caption("Recognized text")
            st.code(page.text, language=None, wrap_lines=True)
            for warning in page.warnings:
                st.warning(warning)
            st.caption("OCR confidence and bounding boxes")
            st.json([line.model_dump() for line in page.lines], expanded=False)
    st.download_button(
        "Download JSON",
        result.model_dump_json(indent=2),
        file_name=f"{Path(result.source).stem}.json",
        mime="application/json",
    )
    with st.expander("Validated JSON"):
        st.json(result.model_dump(), expanded=False)
