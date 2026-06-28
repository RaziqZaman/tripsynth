.PHONY: audit-survey fetch-counts test smoke
export PYTHONPATH := src

audit-survey:
	.venv/bin/python -m tripsynth.cli audit-survey --config configs/default.yaml

fetch-counts:
	.venv/bin/python -m tripsynth.cli fetch-counts --config configs/default.yaml

test:
	.venv/bin/python -m pytest

smoke: audit-survey test
