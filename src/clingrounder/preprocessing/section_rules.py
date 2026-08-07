"""Configurable, offset-safe clinical section rules.

The contract follows the useful part of medspaCy's section architecture:
rules own aliases, semantic categories, optional parent constraints, and scope
limits. Matching remains local and dependency-free so third-party tokenization
cannot become the owner of source offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clingrounder.schema.document import Section

__all__ = [
    "DEFAULT_SECTION_RULE_REGISTRY",
    "RuleBasedSectionDetector",
    "SectionRule",
    "SectionRuleRegistry",
    "split_sections",
]

_LINE_RE = re.compile(r"^.*$", flags=re.MULTILINE)
_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*[\.)]\s*")
_SEPARATOR_RE = re.compile(r"\s*(?::|：)\s*")


@dataclass(frozen=True, slots=True)
class SectionRule:
    """One reusable section-heading rule.

    ``parent_categories`` constrains subsections by semantic category rather
    than by one literal heading. ``max_scope_chars`` is a safety cap for
    malformed documents whose next heading is missing.
    """

    rule_id: str
    title: str
    category: str
    aliases: tuple[str, ...]
    parent_categories: tuple[str, ...] = ()
    parent_required: bool = False
    max_scope_chars: int | None = None

    def __post_init__(self) -> None:
        values = (self.rule_id, self.title, self.category, *self.aliases)
        if any(not value.strip() for value in values):
            raise ValueError("Section rule identifiers, title, category, and aliases are required.")
        normalized_aliases = tuple(alias.casefold().strip() for alias in self.aliases)
        if len(normalized_aliases) != len(set(normalized_aliases)):
            raise ValueError(f"Section rule {self.rule_id!r} has duplicate aliases.")
        if self.parent_required and not self.parent_categories:
            raise ValueError(
                f"Section rule {self.rule_id!r} requires at least one parent category."
            )
        if self.max_scope_chars is not None and self.max_scope_chars < 1:
            raise ValueError("max_scope_chars must be positive when provided.")


@dataclass(frozen=True, slots=True)
class _Heading:
    rule: SectionRule
    heading_start: int
    heading_end: int
    content_start: int
    parent_title: str | None


class SectionRuleRegistry:
    """Validated heading rules ordered for deterministic longest-alias matching."""

    def __init__(self, rules: tuple[SectionRule, ...]) -> None:
        if not rules:
            raise ValueError("At least one section rule is required.")
        rule_ids = [rule.rule_id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Section rule IDs must be unique.")

        aliases: dict[str, str] = {}
        indexed: list[tuple[str, SectionRule]] = []
        for rule in rules:
            for alias in rule.aliases:
                normalized = alias.casefold().strip()
                previous = aliases.get(normalized)
                if previous is not None:
                    raise ValueError(
                        f"Section alias {alias!r} is shared by {previous!r} and {rule.rule_id!r}."
                    )
                aliases[normalized] = rule.rule_id
                indexed.append((normalized, rule))
        self.rules = rules
        self._indexed_aliases = tuple(
            sorted(indexed, key=lambda item: (-len(item[0]), item[1].rule_id))
        )

    def match(self, line: str) -> tuple[SectionRule, int, int] | None:
        """Return rule, heading end, and content start as offsets inside ``line``."""

        prefix = _NUMBER_PREFIX_RE.match(line)
        content_offset = prefix.end() if prefix else 0
        candidate = line[content_offset:]
        stripped_left = len(candidate) - len(candidate.lstrip())
        search_start = content_offset + stripped_left
        normalized = candidate.lstrip().casefold()

        for alias, rule in self._indexed_aliases:
            if not _matches_alias_boundary(normalized, alias):
                continue
            heading_end = search_start + len(alias)
            separator = _SEPARATOR_RE.match(line, heading_end)
            if separator is not None:
                return rule, heading_end, separator.end()
            remainder = line[heading_end:]
            return rule, heading_end, heading_end if remainder.strip() else len(line)
        return None


class RuleBasedSectionDetector:
    """Detect clinical sections while preserving source-coordinate spans."""

    def __init__(self, registry: SectionRuleRegistry) -> None:
        self.registry = registry

    def detect(self, text: str) -> list[Section]:
        headings = self._headings(text)
        if not headings:
            return [Section(title="Document", span=(0, len(text)), text=text)]

        sections: list[Section] = []
        for index, heading in enumerate(headings):
            next_heading = (
                headings[index + 1].heading_start
                if index + 1 < len(headings)
                else len(text)
            )
            content_end = next_heading
            if heading.rule.max_scope_chars is not None:
                content_end = min(
                    content_end,
                    heading.content_start + heading.rule.max_scope_chars,
                )
            content_start = min(heading.content_start, content_end)
            section_text = text[content_start:content_end]
            # INVARIANT: section bodies are views into immutable source text.
            if text[content_start:content_end] != section_text:
                raise AssertionError("Section offsets must address the source text.")
            sections.append(
                Section(
                    title=heading.rule.title,
                    span=(content_start, content_end),
                    text=section_text,
                    category=heading.rule.category,
                    heading_span=(heading.heading_start, heading.heading_end),
                    parent_title=heading.parent_title,
                    rule_id=heading.rule.rule_id,
                )
            )
        return sections

    def _headings(self, text: str) -> list[_Heading]:
        accepted: list[_Heading] = []
        for line_match in _LINE_RE.finditer(text):
            line = line_match.group(0)
            if not line.strip():
                continue
            matched = self.registry.match(line)
            if matched is None:
                continue
            rule, heading_end, content_start = matched
            parent = _nearest_parent(accepted, rule.parent_categories)
            if rule.parent_required and parent is None:
                continue
            accepted.append(
                _Heading(
                    rule=rule,
                    heading_start=line_match.start(),
                    heading_end=line_match.start() + heading_end,
                    content_start=line_match.start() + content_start,
                    parent_title=parent.rule.title if parent is not None else None,
                )
            )
        return _dedupe_headings(accepted)


def split_sections(
    text: str,
    registry: SectionRuleRegistry | None = None,
) -> list[Section]:
    """Detect sections with the canonical section-rule implementation.

    Keeping this small entry point beside the detector prevents a second
    preprocessing module from becoming an accidental public implementation.
    """

    active_registry = registry or DEFAULT_SECTION_RULE_REGISTRY
    return RuleBasedSectionDetector(active_registry).detect(text)


def _nearest_parent(
    headings: list[_Heading],
    parent_categories: tuple[str, ...],
) -> _Heading | None:
    if not parent_categories:
        return None
    allowed = set(parent_categories)
    return next(
        (heading for heading in reversed(headings) if heading.rule.category in allowed),
        None,
    )


def _matches_alias_boundary(text: str, alias: str) -> bool:
    if not text.startswith(alias):
        return False
    if len(text) == len(alias):
        return True
    next_char = text[len(alias)]
    return next_char.isspace() or next_char in {":", "："}


def _dedupe_headings(headings: list[_Heading]) -> list[_Heading]:
    deduped: list[_Heading] = []
    for heading in headings:
        if deduped and heading.heading_start == deduped[-1].heading_start:
            deduped[-1] = heading
        else:
            deduped.append(heading)
    return deduped


DEFAULT_SECTION_RULE_REGISTRY = SectionRuleRegistry(
    (
        SectionRule(
            rule_id="section.medication.preadmission",
            title="Thuốc trước khi nhập viện",
            category="medication_history",
            aliases=(
                "danh sách thuốc trước nhập viện chính xác và đầy đủ",
                "danh sách thuốc trước khi nhập viện chính xác và đầy đủ",
                "danh sách thuốc trước nhập viện",
                "danh sách thuốc trước khi nhập viện",
                "thuốc trước khi nhập viện lần này",
                "thuốc trước khi nhập viện",
            ),
        ),
        SectionRule(
            rule_id="section.history.chronic",
            title="Các bệnh lý mạn tính",
            category="medical_history",
            aliases=(
                "các bệnh lý mãn tính",
                "các bệnh lý mạn tính",
                "bệnh lý mãn tính",
                "bệnh lý mạn tính",
            ),
        ),
        SectionRule(
            rule_id="section.history.medical",
            title="Tiền sử bệnh nội khoa",
            category="medical_history",
            aliases=("tiền sử bệnh nội khoa",),
        ),
        SectionRule(
            rule_id="section.present.history",
            title="Tiền sử bệnh hiện tại",
            category="present_illness",
            aliases=("tiền sử bệnh hiện tại",),
        ),
        SectionRule(
            rule_id="section.present.illness",
            title="Bệnh sử hiện tại",
            category="present_illness",
            aliases=("bệnh sử hiện tại", "lịch sử bệnh hiện tại"),
        ),
        SectionRule(
            rule_id="section.history.general",
            title="Tiền sử bệnh",
            category="medical_history",
            aliases=("tiền sử bệnh",),
        ),
        SectionRule(
            rule_id="section.present.symptoms",
            title="Triệu chứng hiện tại",
            category="symptoms",
            aliases=(
                "các triệu chứng hiện tại",
                "triệu chứng hiện tại",
                "triệu chứng khi đến",
            ),
        ),
        SectionRule(
            rule_id="section.preadmission.events",
            title="Các sự kiện trước khi nhập viện",
            category="preadmission_events",
            aliases=(
                "các sự kiện trước khi nhập viện",
                "diễn biến trước khi nhập viện",
            ),
        ),
        SectionRule(
            rule_id="section.preadmission.status",
            title="Tình trạng ngay trước khi nhập viện",
            category="preadmission_status",
            aliases=("tình trạng ngay trước khi nhập viện",),
        ),
        SectionRule(
            rule_id="section.symptom.characteristics",
            title="Đặc điểm triệu chứng",
            category="symptoms",
            aliases=(
                "đặc điểm triệu chứng khi khám tại khoa cấp cứu",
                "đặc điểm triệu chứng",
            ),
        ),
        SectionRule(
            rule_id="section.assessment.hospital",
            title="Đánh giá tại bệnh viện",
            category="assessment",
            aliases=("đánh giá tại bệnh viện",),
        ),
        SectionRule(
            rule_id="section.examination.results",
            title="Kết quả khám lâm sàng",
            category="examination",
            aliases=("kết quả khám lâm sàng",),
        ),
        SectionRule(
            rule_id="section.laboratory.results",
            title="Kết quả xét nghiệm",
            category="laboratory",
            aliases=("kết quả xét nghiệm",),
        ),
        SectionRule(
            rule_id="section.imaging.results",
            title="Kết quả chẩn đoán hình ảnh",
            category="imaging",
            aliases=("kết quả chẩn đoán hình ảnh",),
        ),
        SectionRule(
            rule_id="section.diagnostics.other",
            title="Các kết quả chẩn đoán khác",
            category="diagnostics",
            aliases=(
                "các kết quả chẩn đoán khác",
                "các phát hiện chẩn đoán khác",
            ),
        ),
        SectionRule(
            rule_id="section.treatment",
            title="Điều trị",
            category="treatment",
            aliases=("các thuốc đã thực hiện", "điều trị"),
        ),
        SectionRule(
            rule_id="section.admission.reason",
            title="Lý do nhập viện",
            category="admission_reason",
            aliases=("lý do nhập viện", "lý do khám bệnh"),
        ),
    )
)
