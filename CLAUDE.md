# dakera-py

Python SDK for the Dakera AI agent memory platform (PyPI package: `dakera`).

## Key Commands
```bash
uv sync                       # Install dependencies (preferred)
uv run pytest tests/          # Run tests
uv run ruff check src/        # Lint
uv run ruff format src/       # Format
uv build                      # Build wheel + sdist → dist/
uv publish                    # Publish to PyPI
```

## Architecture
- `src/dakera/` — Main package:
  - `client.py` — Synchronous DakeraClient (requests-based)
  - `async_client.py` — Async AsyncDakeraClient (httpx-based)
  - `models.py` — All request/response types (Pydantic)
  - `exceptions.py` — DakeraError and subclasses
- `tests/` — Pytest integration tests (requires live server or mock)
- `examples/` — Usage examples by use case

## Conventions
- Python 3.10+; ruff line-length 100; imports sorted (I001 enforced)
- CI matrix: default 3.9 + 3.12; `full-matrix` label adds 3.10 + 3.11
- Version in pyproject.toml matches server version (e.g., 0.9.13)
- SDK batch: all 4 SDKs (py, js, rs, go) sync together after a server API change
