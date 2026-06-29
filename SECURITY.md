# Security Policy

## Supported Versions

This repository is an early research prototype. Security fixes target the default branch unless a
release branch is explicitly maintained.

## Reporting a Vulnerability

Do not open public issues for vulnerabilities, exposed credentials, private clinical data, or dataset
license leaks.

Report privately to the maintainers through the repository security advisory channel when available,
or contact the project owner directly.

Include:

- A concise description of the issue.
- Steps to reproduce.
- Affected files, commands, or outputs.
- Whether private data, credentials, or licensed datasets may be exposed.

## Data Handling

- Do not commit PHI, private EHR notes, restricted benchmark data, or credentialed dataset exports.
- Keep local datasets under ignored paths such as `data/raw/` and `data/processed/`.
- Use synthetic data for tests and examples.
- Remove generated outputs that contain private text before sharing logs or PRs.
