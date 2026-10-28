.PHONY: test lint mvp

test:
	pytest -q

lint:
	ruff check .

mvp:
	python scripts/build_mvp_fixtures.py
	toklab run experiment --config configs/experiments/mvp_py.yaml
