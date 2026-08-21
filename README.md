# marine_recruitment_data

Data collection and analysis for marine recruitment.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Layout

```
src/marine_recruitment_data/   package code
tests/                         pytest tests
data/raw/                      source data, never edited in place (gitignored)
data/processed/                derived data (gitignored)
notebooks/                     exploratory notebooks
```

## Tests

```bash
pytest
```
