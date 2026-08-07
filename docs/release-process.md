# Release And Deployment

ClinGrounder uses GitHub Actions as the release boundary. A release tag is the single deployment
input: it runs the verification suite, builds the Python artifacts, publishes to PyPI, and creates a
GitHub Release with the matching section from `CHANGELOG.md`.

## One-Time Repository Setup

Configure a PyPI Trusted Publisher for:

- PyPI project: `clingrounder`
- GitHub owner/repository: `damminhtien/clingrounder`
- Workflow: `release.yml`
- Environment: `pypi`

The workflow uses GitHub's OIDC token. No `PYPI_TOKEN`, long-lived secret, or token in the repository
is required. The `pypi` environment should require an approval for production releases. The GitHub
Release is created independently of the PyPI job, so a PyPI configuration problem cannot hide the
versioned source and wheel artifacts on GitHub.

## Release Checklist

1. Update `CHANGELOG.md` with a section whose heading exactly matches the package version.
2. Update `version` in `pyproject.toml`.
3. Run the local gates:

   ```bash
   uv sync --frozen --extra dev --extra data
   uv run ruff check .
   uv run mypy src
   uv run pytest -o addopts='' -m "not private and not model and not benchmark" tests
   uv run clingrounder release audit --root .
   uv build --wheel --sdist --out-dir dist
   uv run --with twine twine check dist/*
   ```

4. Commit the version and changelog changes.
5. Create and push an annotated tag matching the package version:

   ```bash
   git tag -a v0.1.0a1 -m "Release ClinGrounder 0.1.0a1"
   git push origin v0.1.0a1
   ```

For an existing tag whose first run failed before publishing, use the GitHub Actions
`workflow_dispatch` input `release_tag` instead of recreating the tag.

The workflow rejects a tag that does not equal `v<project.version>`. It also fails if the matching
changelog section is missing, if release validation fails, or if package metadata is invalid.

## Deployment Outputs

- PyPI: `https://pypi.org/project/clingrounder/`
- GitHub Releases: repository Releases page, with wheel and source distribution attached.
- CI artifacts: build and security reports remain available from the workflow run.

Large corpora, model weights, licensed terminology archives, and generated research runs are not
deployed through this workflow. Their content-addressed release locks and provenance remain managed
by the data-mining release process.

## Rollback

PyPI artifacts are immutable. Roll back consumers by pinning a previously published version; do not
overwrite a release. For a correction, increment the version, document the change, and create a new
tag. A GitHub Release can be marked as a pre-release while the public API is still in alpha.
