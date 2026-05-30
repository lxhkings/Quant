from pathlib import Path

from quant.config import data_lake_root


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_DATA_LAKE_ROOT", str(tmp_path))
    assert data_lake_root() == tmp_path


def test_default_points_to_trendspec(monkeypatch):
    monkeypatch.delenv("QUANT_DATA_LAKE_ROOT", raising=False)
    root = data_lake_root()
    assert root.name == "data_lake"
    assert "TrendSpec" in str(root)
