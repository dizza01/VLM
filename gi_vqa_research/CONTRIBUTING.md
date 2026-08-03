# Contributing

Contributions should preserve the project's reproducibility and test-isolation
contracts.

1. Create a focused branch from a clean checkout.
2. Do not commit datasets, images, model weights, predictions, credentials or
   raw run directories.
3. Add or update tests for behavioural changes.
4. Run `make publication-check`, `make test` under Python 3.11,
   `python -m ruff check scripts/publication_check.py`, and
   `python -m ruff check . --select E9,F63,F7,F82`.
5. Update versioned protocol or profile identifiers when a scientific
   parameter changes. Refresh bound hashes explicitly and disclose the change.
6. Never access the official test partition from training, development or
   protocol-validation code.

Security concerns should follow [SECURITY.md](SECURITY.md), not a public issue.
