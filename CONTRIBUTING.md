# Contributing to Dakera

Thank you for your interest in contributing to Dakera! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add your feature'`)
6. Push to your branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## Development Setup

**Prerequisites:** Python 3.8+

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy src/
```

## Code Style

- Follow the existing code style and conventions
- Write clear, descriptive commit messages
- Add tests for new functionality
- Update documentation as needed

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include a clear description of what changed and why
- Ensure all CI checks pass
- Link relevant issues

## Reporting Issues

Use the [Bug Report](https://github.com/Dakera-AI/dakera-py/issues/new?template=bug_report.md) template to report bugs. Please include:
- Python version, operating system, and dakera package version
- Steps to reproduce the issue
- Expected vs actual behavior, with full error traceback

Have a feature idea? Use the [Feature Request](https://github.com/Dakera-AI/dakera-py/issues/new?template=feature_request.md) template.

## Security Vulnerabilities

**Do not open public issues for security vulnerabilities.** See [SECURITY.md](.github/SECURITY.md) for responsible disclosure instructions — report via [GitHub Security Advisories](https://github.com/dakera-ai/dakera-py/security/advisories/new).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
