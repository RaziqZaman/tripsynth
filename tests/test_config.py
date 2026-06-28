from pathlib import Path

from tripsynth.config import flatten_study_counties, load_config, resolve_path


def test_default_config_loads():
    config = load_config("configs/default.yaml")
    assert config["validation"]["default_mode"] == "spatial_aadt_proxy"
    assert "11001" in flatten_study_counties(config)


def test_resolve_path_prefers_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("survey.csv").write_text("a\n1\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "default.yaml"
    config_path.parent.mkdir()
    config_path.write_text("survey: {}\n", encoding="utf-8")
    config = load_config(config_path)
    assert resolve_path(config, "survey.csv") == tmp_path / "survey.csv"
