from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from ocr_information_extraction.cli import main
from ocr_information_extraction.schemas import DocumentResult, PageResult, Receipt


def test_cli_invalid_input_and_missing_model(capsys, tmp_path):
    assert main(["extract", str(tmp_path / "missing.png")]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert main(["extract", "receipt.png", "--method", "svm"]) == 1
    assert "requires a trained model" in capsys.readouterr().err
    assert main(["extract", "receipt.png", "--method", "svm", "--model", "missing.joblib"]) == 1
    assert "Model not found" in capsys.readouterr().err


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "prepare" in capsys.readouterr().out


def test_streamlit_initial_view():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Receipt OCR & information extraction"
    assert app.selectbox[-1].options == ["regex"]


def test_streamlit_result_view():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    result = DocumentResult(
        source="synthetic.png",
        language="eng",
        psm=6,
        pdf_dpi=150,
        preprocessing={},
        tesseract_version="test",
        elapsed_seconds=0.1,
        pages=[
            PageResult(
                page=1,
                fields=Receipt(total=12.5),
                text="TOTAL 12.50",
                method="regex",
                lines=[],
                elapsed_seconds=0.1,
            )
        ],
    )
    image = Image.new("RGB", (20, 20), "white")
    app.session_state["analysis"] = {"result": result.model_dump(), "previews": [(image, image)]}
    app.run(timeout=30)
    assert not app.exception
    assert app.code[0].value == "TOTAL 12.50"
    assert len(app.table) == 1
