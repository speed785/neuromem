# Contributing to neuromem

Thanks for contributing to neuromem. This project maintains strict quality and reliability standards, including 100% test coverage for the Python package.

## Development setup

### Prerequisites

- Python 3.10+
- Node.js 18+

### Clone and install

```bash
git clone git@github.com:speed785/neuromem.git
cd neuromem

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,openai,anthropic,langchain]"

cd typescript
npm ci
cd ..
```

## Running checks

Run all checks before opening a pull request.

```bash
ruff check neuromem tests --ignore F401
mypy neuromem --ignore-missing-imports
pytest --cov=neuromem --cov-report=term-missing --cov-fail-under=100

cd typescript
npm run build
cd ..
```

## Test and coverage policy

- Python changes must keep coverage at 100%.
- New behavior must include tests.
- Bug fixes must include a regression test.

## Pull request guidelines

- Keep PRs focused and atomic.
- Use clear commit messages.
- Update `CHANGELOG.md` for user-facing changes.
- Ensure CI is green before requesting review.
- Include rationale and testing notes in the PR description.

## Reporting issues

- Use the bug report template for defects.
- Use the feature request template for enhancements.
- For vulnerabilities, follow `SECURITY.md`.
