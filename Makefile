.PHONY: help train test test-full check format smoke
help:
	@echo "train / test / test-full / check / format / smoke"
train:
	uv run python src/train.py
test:
	uv run pytest -m 'not integration'
test-full:
	uv run pytest
check:
	uv run ruff check .
	uv run ruff format --check .
format:
	uv run pre-commit run --all-files
smoke:
	uv run python src/train.py experiment=smoke
