from __future__ import annotations

import csv
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.dictionaries.icd10_sources import ICD10_VN_TT06_2026_SOURCE_ID


_ICD10_CODE_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")
_STT_X_RANGE = (15.0, 30.0)
_CODE_X_RANGE = (468.0, 488.0)
_ENGLISH_NAME_X_RANGE = (505.0, 547.0)
_VIETNAMESE_NAME_X_RANGE = (591.0, 640.0)


@dataclass(frozen=True)
class TT06ExtractedRow:
    stt: str
    code: str
    official_name_vi: str
    official_name_en: str
    page: int

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "official_name_vi": self.official_name_vi,
            "official_name_en": self.official_name_en,
            "source_id": ICD10_VN_TT06_2026_SOURCE_ID,
            "source": ICD10_VN_TT06_2026_SOURCE_ID,
            "page": self.page,
            "stt": self.stt,
        }
        parent_code = _parent_code(self.code)
        if parent_code is not None:
            payload["parent_code"] = parent_code
        return payload


def extract_tt06_rows_from_pdf(
    pdf_path: str | Path,
    *,
    tsv_path: str | Path | None = None,
) -> list[TT06ExtractedRow]:
    effective_tsv_path = Path(tsv_path) if tsv_path is not None else Path(pdf_path).with_suffix(".tsv")
    run_pdftotext_tsv(pdf_path, effective_tsv_path)
    return extract_tt06_rows_from_tsv(effective_tsv_path)


def run_pdftotext_tsv(pdf_path: str | Path, tsv_path: str | Path) -> None:
    output_path = Path(tsv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftotext",
            "-tsv",
            "-enc",
            "UTF-8",
            str(pdf_path),
            str(output_path),
        ],
        check=True,
    )


def extract_tt06_rows_from_tsv(tsv_path: str | Path) -> list[TT06ExtractedRow]:
    lines = _load_word_lines(tsv_path)
    rows: list[TT06ExtractedRow] = []
    current: _RowBuilder | None = None
    for (page, _y), words in sorted(lines.items()):
        sorted_words = sorted(words, key=lambda word: word[0])
        stt = _first_word_in_range(sorted_words, _STT_X_RANGE, str.isdigit)
        code = _first_word_in_range(sorted_words, _CODE_X_RANGE, _is_icd10_code)
        if code is not None:
            if current is not None:
                maybe_row = current.build()
                if maybe_row is not None:
                    rows.append(maybe_row)
            # The official PDF merges STT cells across many ICD rows. Requiring STT on every
            # line silently dropped valid codes; the tightly bounded code column is sufficient.
            current = _RowBuilder(stt=stt or "", code=code, page=page)
        if current is not None:
            current.official_name_en.extend(_words_in_range(sorted_words, _ENGLISH_NAME_X_RANGE))
            current.official_name_vi.extend(_words_in_range(sorted_words, _VIETNAMESE_NAME_X_RANGE))
    if current is not None:
        maybe_row = current.build()
        if maybe_row is not None:
            rows.append(maybe_row)
    return rows


def build_tt06_manifest(
    *,
    rows: Sequence[TT06ExtractedRow],
    source_pdf: str | Path,
    output_jsonl: str | Path,
    output_tsv: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "source_id": ICD10_VN_TT06_2026_SOURCE_ID,
        "source_pdf": str(source_pdf),
        "output_jsonl": str(output_jsonl),
        "output_tsv": str(output_tsv) if output_tsv is not None else None,
        "rows": len(rows),
        "unique_codes": len({row.code for row in rows}),
        "first_code": rows[0].code if rows else None,
        "last_code": rows[-1].code if rows else None,
    }


@dataclass
class _RowBuilder:
    stt: str
    code: str
    page: int
    official_name_vi: list[str]
    official_name_en: list[str]

    def __init__(self, *, stt: str, code: str, page: int) -> None:
        self.stt = stt
        self.code = code
        self.page = page
        self.official_name_vi = []
        self.official_name_en = []

    def build(self) -> TT06ExtractedRow | None:
        official_name_vi = _clean_words(self.official_name_vi)
        if not official_name_vi:
            return None
        return TT06ExtractedRow(
            stt=self.stt,
            code=self.code,
            official_name_vi=official_name_vi,
            official_name_en=_clean_words(self.official_name_en),
            page=self.page,
        )


def _load_word_lines(tsv_path: str | Path) -> dict[tuple[int, int], list[tuple[float, str]]]:
    csv.field_size_limit(sys.maxsize)
    lines: dict[tuple[int, int], list[tuple[float, str]]] = defaultdict(list)
    with Path(tsv_path).open("r", encoding="utf-8", newline="") as handle:
        # pdftotext emits raw clinical prose, including unmatched double quotes. TSV has no
        # quoting contract, so standard CSV quote handling can swallow many subsequent pages.
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row.get("level") != "5":
                continue
            text = str(row.get("text", "")).strip()
            if not text or text.startswith("###"):
                continue
            page = int(str(row["page_num"]))
            y = round(float(str(row["top"])) / 3.0) * 3
            lines[(page, y)].append((float(str(row["left"])), text))
    return lines


def _first_word_in_range(
    words: Sequence[tuple[float, str]],
    x_range: tuple[float, float],
    predicate: Callable[[str], bool],
) -> str | None:
    for left, text in words:
        if _in_range(left, x_range) and predicate(text):
            return text
    return None


def _words_in_range(words: Iterable[tuple[float, str]], x_range: tuple[float, float]) -> Iterator[str]:
    for left, text in words:
        if _in_range(left, x_range):
            yield text


def _in_range(value: float, x_range: tuple[float, float]) -> bool:
    return x_range[0] <= value < x_range[1]


def _is_icd10_code(value: str) -> bool:
    return bool(_ICD10_CODE_RE.match(value))


def _clean_words(words: Iterable[str]) -> str:
    return " ".join(" ".join(words).split()).strip()


def _parent_code(code: str) -> str | None:
    if "." not in code:
        return None
    return code.split(".", maxsplit=1)[0]
