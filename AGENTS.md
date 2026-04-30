# Project Instructions

## Python Environment
- This project uses `uv` exclusively.
- Never use `pip`, `pip3`, `python -m pip`, `poetry`, or `conda` for dependency management.
- Prefer project-root execution unless stated otherwise.

## Dependency Management
- Add runtime dependencies with: `uv add <package>`
- Add dev dependencies with: `uv add --dev <package>`
- Remove dependencies with: `uv remove <package>`
- Sync environment with: `uv sync`

## Running Python
- Run scripts with: `uv run <script>.py`
- Run modules/tools with: `uv run <command>`
- Run tests with: `uv run pytest`
- Run lint/format with: `uv run ruff check .` and `uv run ruff format .`

## Safety
- Before introducing a new package, explain why and then use `uv add`.
- Do not use global Python or system site-packages.