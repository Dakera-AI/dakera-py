---
description: Python SDK conventions for dakera-py
globs: "*.py"
---

# Python SDK Conventions

- Match server API 1:1 — every public endpoint needs a client method
- Use httpx for HTTP client (async and sync)
- Export all public types from __init__.py
- Version in pyproject.toml must match latest server version
- All new methods need unit tests with mock responses
- Use RetryConfig for automatic retries — never retry in business logic
- Rate-limit headers (X-RateLimit-*) must be exposed via RateLimitHeaders
