# 0004: Open-Source Repository Hygiene

## Status

Accepted.

## Context

The repository is intended to be reusable as a research-grade prototype. External contributors need
clear licensing, contribution rules, security/data-handling guidance, CI, issue templates, and local
quality commands.

## Decision

Use a standard open-source wrapper around the Python package:

- MIT license.
- Contribution, code of conduct, security, support, changelog, and citation files.
- GitHub issue templates, PR template, CI, and Dependabot config.
- EditorConfig, Git attributes, Makefile shortcuts, and pre-commit hooks.

## Consequences

- Contributors get a predictable workflow.
- Private clinical data and restricted datasets are explicitly out of scope for public issues and
  commits.
- CI checks the same core gates used locally: ruff, mypy, pytest, pipeline smoke, and prediction
  validation.
