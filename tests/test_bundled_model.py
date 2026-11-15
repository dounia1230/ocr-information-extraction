"""Check that a fresh checkout can use the shipped artifact without training data."""

from pathlib import Path

from ocr_information_extraction.ml import extract_svm, load_model
from ocr_information_extraction.schemas import LABELS


def test_bundled_model_loads_and_extracts_without_preparation(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "models" / "receipt-svm.joblib"
    monkeypatch.chdir(tmp_path)
    assert not Path("data").exists()
    model = load_model(path)
    assert set(model["pipeline"].classes_) == set(LABELS)
    result = extract_svm(
        "EXAMPLE SHOP\n12 MAIN STREET\n12345 TOWN\nDate: 12/10/2030\nItem 20.00\nTOTAL RM 25.50",
        model,
    )
    assert result.date == "2030-10-12"
    assert result.total == 25.5
