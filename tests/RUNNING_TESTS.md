# Running Tests

## Prerequisites

The tests require the package to be installed in editable mode. This allows tests to import modules directly without modifying `sys.path`.

### Install Package in Editable Mode

```bash
# Install with dev dependencies
uv sync --extra dev

# Or if using pip
pip install -e .[dev]
```

This installs the package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

## Running Tests

```bash
# Run all LMCleaner tests
uv run pytest tests/test_lmcleaner_*.py -v

# Run specific test file
uv run pytest tests/test_lmcleaner_core.py -v

# Run with coverage
uv run pytest tests/test_lmcleaner_*.py --cov=src/trainer/unlearn --cov=src/trainer/training_logger --cov-report=html

# Run all tests
make test
```

## Test Structure

Tests use shared fixtures defined in `tests/conftest.py`:
- `simple_model`: A simple neural network for testing
- `batch_data`: Sample batch data
- `temp_log_dir`: Temporary directory for test outputs

This eliminates duplication and ensures consistency across test files.

## Expected Results

- **69 tests passing** (97%)
- **2 tests skipped** (3%) - Known code bugs documented in TEST_FAILURE_ANALYSIS.md

## Troubleshooting

### ImportError: No module named 'trainer'

**Solution**: Install the package in editable mode:
```bash
uv sync --extra dev
```

### Tests fail with "permission denied"

**Solution**: Set TMPDIR environment variable:
```bash
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
make test
```
