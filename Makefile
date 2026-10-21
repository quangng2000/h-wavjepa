.PHONY: test check smoke
test:
	uv run pytest
check:
	uv run ruff check .
	uv run ruff format --check .
smoke:
	uv run python -m src.train experiment=smoke
