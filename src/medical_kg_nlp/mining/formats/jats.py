"""Offset-bearing rendering records for Journal Article Tag Suite documents."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

__all__ = ["RenderedJatsArticle", "RenderedJatsBlock", "render_jats_article"]


@dataclass(frozen=True)
class RenderedJatsBlock:
    """One source-structured title or paragraph projected into rendered article text."""

    kind: str
    span: tuple[int, int]
    section_path: tuple[str, ...]
    section_type: str
    text_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "span": list(self.span),
            "section_path": list(self.section_path),
            "section_type": self.section_type,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True)
class RenderedJatsArticle:
    """Immutable article text plus exact block provenance and bibliographic metadata."""

    article_id: str
    text: str
    blocks: tuple[RenderedJatsBlock, ...]
    article_type: str
    journal_title: str
    publication_year: str
    keywords: tuple[str, ...]
    subjects: tuple[str, ...]


@dataclass(frozen=True)
class _PendingBlock:
    text: str
    kind: str
    section_path: tuple[str, ...]
    section_type: str


def render_jats_article(root: ET.Element, *, fallback_id: str) -> RenderedJatsArticle:
    """Render the same article text as parser revision 1 while retaining block spans."""

    article_id = _first_text(root, ".//article-id[@pub-id-type='pmc']")
    article_id = article_id or _first_text(root, ".//article-id") or fallback_id
    parent_by_node = {child: parent for parent in root.iter() for child in parent}
    pending: list[_PendingBlock] = []
    title = root.find(".//article-title")
    if title is not None and (text := node_text(title)):
        pending.append(_PendingBlock(text, "article_title", (), ""))
    for path, container_kind in ((".//abstract", "abstract"), (".//body", "body")):
        container = root.find(path)
        if container is None:
            continue
        for node in container.iter():
            local_name = _local_name(node.tag)
            if local_name not in {"title", "p"}:
                continue
            text = node_text(node)
            if not text:
                continue
            section_path, section_type = _section_context(
                node,
                container=container,
                parent_by_node=parent_by_node,
                container_kind=container_kind,
            )
            kind = "section_title" if local_name == "title" else "paragraph"
            pending.append(_PendingBlock(text, kind, section_path, section_type))

    # INVARIANT: revision 1 removed only adjacent identical blocks. Preserve that exact policy so
    # richer provenance cannot alter source text or invalidate previously reviewed offsets.
    deduplicated: list[_PendingBlock] = []
    for block in pending:
        if not deduplicated or deduplicated[-1].text != block.text:
            deduplicated.append(block)
    text, blocks = _render_blocks(deduplicated)
    article = root.find(".//article") if _local_name(root.tag) != "article" else root
    article_type = "" if article is None else str(article.get("article-type", "")).strip()
    return RenderedJatsArticle(
        article_id=article_id,
        text=text,
        blocks=blocks,
        article_type=article_type,
        journal_title=_first_text(root, ".//journal-title"),
        publication_year=_publication_year(root),
        keywords=_unique_text(root.findall(".//kwd")),
        subjects=_unique_text(root.findall(".//subj-group/subject")),
    )


def node_text(node: ET.Element) -> str:
    """Collapse XML inline markup without changing word order."""

    return " ".join("".join(node.itertext()).split())


def _render_blocks(
    blocks: list[_PendingBlock],
) -> tuple[str, tuple[RenderedJatsBlock, ...]]:
    parts: list[str] = []
    rendered: list[RenderedJatsBlock] = []
    cursor = 0
    for index, block in enumerate(blocks):
        if index:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(block.text)
        cursor += len(block.text)
        rendered.append(
            RenderedJatsBlock(
                kind=block.kind,
                span=(start, cursor),
                section_path=block.section_path,
                section_type=block.section_type,
                text_sha256=hashlib.sha256(block.text.encode("utf-8")).hexdigest(),
            )
        )
    return "".join(parts), tuple(rendered)


def _section_context(
    node: ET.Element,
    *,
    container: ET.Element,
    parent_by_node: dict[ET.Element, ET.Element],
    container_kind: str,
) -> tuple[tuple[str, ...], str]:
    sections: list[ET.Element] = []
    current = node
    while current is not container:
        parent = parent_by_node.get(current)
        if parent is None:
            break
        if _local_name(parent.tag) == "sec":
            sections.append(parent)
        current = parent
    sections.reverse()
    titles = tuple(title for section in sections if (title := _direct_section_title(section)))
    if container_kind == "abstract":
        titles = ("Abstract", *titles)
    section_type = next(
        (
            value
            for section in reversed(sections)
            if (value := str(section.get("sec-type", "")).strip())
        ),
        container_kind,
    )
    return titles, section_type


def _direct_section_title(section: ET.Element) -> str:
    title = next((child for child in section if _local_name(child.tag) == "title"), None)
    return "" if title is None else node_text(title)


def _first_text(root: ET.Element, path: str) -> str:
    node = root.find(path)
    return "" if node is None else node_text(node)


def _publication_year(root: ET.Element) -> str:
    for publication_type in ("epub", "ppub", "collection"):
        node = root.find(f".//pub-date[@pub-type='{publication_type}']/year")
        if node is not None and (value := node_text(node)):
            return value
    return _first_text(root, ".//pub-date/year")


def _unique_text(nodes: list[ET.Element]) -> tuple[str, ...]:
    values = {text for node in nodes if (text := node_text(node))}
    return tuple(sorted(values, key=lambda value: (value.casefold(), value)))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
