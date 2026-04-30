---
name: sdk-release
description: Release the Dakera Python SDK. Use when publishing a new version to PyPI.
disable-model-invocation: true
allowed-tools: Bash(gh *) Bash(python *) Bash(pip *)
---

## Python SDK Release

### Pre-release checks
```bash
python -m pytest
python -m ruff check .
python -m mypy dakera/
```

### Version bump
Update version in `pyproject.toml` under `[project].version`.

### Release process
1. Update `CHANGELOG.md`
2. Commit: `git commit -m "chore: bump to vX.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin main --tags`
5. Release workflow auto-publishes to PyPI

### Batching rules
- All 4 SDKs (py, js, rs, go) sync in a single coordinated batch
- Do NOT release for a single trivial change — batch until 2+ changes or security fix
- SDK patch releases only when threshold is met
