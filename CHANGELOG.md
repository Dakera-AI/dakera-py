# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-03-22

### Security
- Bumped `urllib3` from 2.2.3 → 2.6.3 (via `requests` transitive dependency) — resolves 5 CVEs:
  - **CVE-2024-37891** (HIGH): urllib3 forwarding `Authorization` headers to third-party redirects
  - Two additional HIGH-severity decompression vulnerability CVEs
  - Two MEDIUM-severity CVEs
- Dropped Python 3.8 support (EOL October 2024); minimum supported version is now Python 3.9

## [0.7.0] - 2026-03-22

### Added
- Batch recall endpoint: `POST /v1/memories/recall/batch` — `recall_batch()` / `async_recall_batch()` methods
- Batch forget endpoint: `DELETE /v1/memories/forget/batch` — `forget_batch()` / `async_forget_batch()` methods
- Rate-limit response headers exposed: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## [0.6.2] - 2026-03-21

### Added
- `CrossAgentNetworkResponse.node_count` field — reflects the `node_count` field added in
  dakera server v0.6.2 (PR #26). Previously the field was silently ignored.
- SSE endpoints now support `?api_key=<key>` query-parameter authentication in addition to
  the `Authorization: Bearer` header. Useful when constructing streaming URLs for clients that
  cannot send custom headers (e.g. browser-native `EventSource`).

## [0.2.0] - 2026-03-19

### Changed
- Version aligned with Go and Rust SDKs (v0.2.0 parity)

## [0.1.0] - 2025-03-15

### Added
- Initial release of Dakera Python SDK
- Synchronous client with full API coverage
- Vector operations: upsert, query, fetch, delete
- Namespace management
- Full-text search support
- Agent memory operations
- Session management
- Knowledge graph operations
- Inference (auto-embedding) support
- Typed models for all API responses
- Comprehensive test suite
- Example scripts
