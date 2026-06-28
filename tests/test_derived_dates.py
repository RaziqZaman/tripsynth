import yaml

from tripsynth.config import load_config
from tripsynth.preprocessing.survey_audit import run_survey_audit


def test_audit_derives_dates_from_week_after_2017_and_iso_dow(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    survey = tmp_path / "survey.csv"
    survey.write_text(
        "o,d,week,dow,dep\n"
        "11001000100,11001000200,40,2,480\n"
        "11001000100,11001000200,131,4,500\n",
        encoding="utf-8",
    )
    payload = {
        "survey": {
            "path": "survey.csv",
            "schema": {
                "origin_tract": "o",
                "destination_tract": "d",
                "survey_week": "week",
                "day_of_week": "dow",
                "departure_time": "dep",
            },
            "temporal_derivation": {
                "derive_trip_date_from_week_dow": True,
                "reference_date": "2017-01-01",
                "week_field": "survey_week",
                "day_of_week_field": "day_of_week",
                "week_index_base": 0,
            },
        },
        "study_area": {"counties": {"dc": ["11001"]}},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    result = run_survey_audit(load_config(config_path))
    assert result.metadata["first_survey_date"] == "2017-10-10"
    assert result.metadata["last_survey_date"] == "2019-07-11"
    assert result.metadata["covered_months_by_year"] == {"2017": [10], "2019": [7]}
