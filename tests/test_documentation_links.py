"""Keep the public quickstart documents free of broken local links."""

from __future__ import annotations

import re
from pathlib import Path


_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
_PUBLIC_DOCS = (Path("README.md"), Path("README_VI.md"))


def test_public_readmes_have_existing_local_links() -> None:
    """Catch stale paths when a public module or document is renamed."""

    missing: list[str] = []
    for document in _PUBLIC_DOCS:
        source = document.read_text(encoding="utf-8")
        for target in _MARKDOWN_LINK.findall(source):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            local_target = target.split("#", 1)[0]
            if not local_target:
                continue
            resolved = (document.parent / local_target).resolve()
            if not resolved.exists():
                missing.append(f"{document}: {target}")

    assert not missing, "Broken local links:\n" + "\n".join(sorted(missing))
