# Tests for DeepSeek API Client

This directory contains unit tests for the DeepSeek API client library.

## Running Tests

To run the tests, first install the development dependencies:

```bash
pip install -e .[dev]
```

Then run pytest:

```bash
pytest tests/ -v
```

For test coverage:

```bash
pytest tests/ --cov=src/deepseek_api
```

## Test Structure

- `conftest.py`: Shared pytest fixtures
- `test_api.py`: Tests for the main `DeepSeekAPI` class
- `test_pow_solve.py`: Tests for the `POWSolver` class (Proof of Work)
- `test_wasm_download.py`: Tests for the WASM download utility
- `README.md`: This file

## Notes

- External HTTP calls are mocked to avoid hitting the real DeepSeek API.
- The WASM solver tests use mocks to avoid requiring actual WebAssembly execution.