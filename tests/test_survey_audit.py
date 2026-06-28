from pathlib import Path

import yaml

from tripsynth.config import load_config
from tripsynth.preprocessing.survey_audit import run_survey_audit


def write_config(path: Path, survey_path: str, schema: dict):
    payload = {
        "survey": {"path": survey_path, "schema": schema},
        "study_area": {"counties": {"dc": ["11001"]}},
        "validation": {"default_mode": "spatial_aadt_proxy"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_audit_with_no_date_field_marks_temporal_unsupported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "o,d,dow,dep,weight\n11001000100,11001000200,1,480,2.0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        "survey.csv",
        {
            "origin_tract": "o",
            "destination_tract": "d",
            "day_of_week": "dow",
            "departure_time": "dep",
            "weight": "weight",
        },
    )
    result = run_survey_audit(load_config(config_path))
    assert result.metadata["calendar_coverage_status"] == "unknown"
    assert result.metadata["usable_for_calendar_temporal_validation"] is False
    assert result.metadata["usable_for_spatial_aadt_proxy_validation"] is True
    assert Path("data/metadata/survey_audit.json").exists()


def test_audit_with_year_month_fields_records_partial_coverage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "o,d,year,month,dep\n1,2,2018,1,480\n1,2,2018,2,500\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    write_config(
        config_path,
        "survey.csv",
        {
            "origin_tract": "o",
            "destination_tract": "d",
            "survey_year": "year",
            "survey_month": "month",
            "departure_time": "dep",
        },
    )
    result = run_survey_audit(load_config(config_path))
    assert result.metadata["full_year_coverage_by_year"] == {"2018": False}
    assert result.metadata["usable_for_calendar_temporal_validation"] is True
