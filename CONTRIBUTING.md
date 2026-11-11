# Contributing

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Before opening a PR

```bash
pytest -q -W error
```

Tests marked `network` are skipped by default (`pytest.ini`); only run them
if your change touches Hub downloads.

## Guidelines

- Keep changes scoped — a bug fix shouldn't carry drive-by refactors.
- Match the existing style in the file you're editing.
- New behavior needs a test; bug fixes should include a test that fails
  without the fix.
- For non-trivial changes, open an issue first to discuss the approach.

## Pull requests

- Describe what the change accomplishes and why, not a file-by-file diff.
- Link any related issue.
- Make sure `pytest -q -W error` passes before requesting review.
