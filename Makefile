.PHONY: test lint mvp

test:
	pytest -q

lint:
	ruff check .

mvp:
	@echo "not yet"
