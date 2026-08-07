"""Link structured DailyMed products to RxNorm with two-source agreement.

One SPL label can contain many products, so an exact ``set_id + version`` lookup
alone is insufficient.  Automatic links require a unique intersection between
the official DailyMed SPL mapping and the independent RxNorm NDC product index.
Every non-unique or disagreeing record remains visible in the decision manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clingrounder.mining.io import iter_documents, write_json, write_jsonl
from clingrounder.mining.mappings.dailymed_rxnorm import (
    DailyMedRxNormMappingRepository,
)
from clingrounder.mining.mappings.rxnorm_ndc import (
    RxNormNdcRepository,
    normalize_ndc_product_prefix,
)
from clingrounder.mining.records import MinedDocument
from clingrounder.utils.hashing import sha256_file

__all__ = ["DailyMedProductRxNormLink", "link_dailymed_products_to_rxnorm"]


@dataclass(frozen=True)
class DailyMedProductRxNormLink:
    """One high-precision source record linked to an active RxNorm concept."""

    link_id: str
    document_id: str
    span: tuple[int, int]
    text: str
    set_id: str
    spl_version: str
    ndc: str
    ndc_product_prefix: str
    rxcui: str
    rxstrings: tuple[str, ...]
    rxttys: tuple[str, ...]
    dailymed_source_version: str
    dailymed_mapping_version: str
    dailymed_mapping_sha256: str
    rxnorm_source_version: str
    rxnorm_source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "link_id": self.link_id,
            "document_id": self.document_id,
            "span": list(self.span),
            "text": self.text,
            "set_id": self.set_id,
            "spl_version": self.spl_version,
            "ndc": self.ndc,
            "ndc_product_prefix": self.ndc_product_prefix,
            "rxcui": self.rxcui,
            "rxstrings": list(self.rxstrings),
            "rxttys": list(self.rxttys),
            "evidence": "exact_set_version_ndc_intersection",
            "dailymed_source_version": self.dailymed_source_version,
            "dailymed_mapping_version": self.dailymed_mapping_version,
            "dailymed_mapping_sha256": self.dailymed_mapping_sha256,
            "rxnorm_source_version": self.rxnorm_source_version,
            "rxnorm_source_sha256": self.rxnorm_source_sha256,
        }


def link_dailymed_products_to_rxnorm(
    documents_path: str | Path,
    *,
    dailymed_mapping_index_path: str | Path,
    rxnorm_ndc_index_path: str | Path,
    links_path: str | Path,
    decisions_path: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Materialize deterministic exact links and all withheld decisions."""

    documents = Path(documents_path)
    mapping_index = Path(dailymed_mapping_index_path)
    ndc_index = Path(rxnorm_ndc_index_path)
    links_target = Path(links_path)
    decisions_target = Path(decisions_path)
    report_target = Path(report_path)
    for target in (links_target, decisions_target, report_target):
        target.parent.mkdir(parents=True, exist_ok=True)

    mapping = DailyMedRxNormMappingRepository(mapping_index)
    ndc = RxNormNdcRepository(ndc_index)
    mapping_metadata = mapping.metadata
    ndc_metadata = ndc.metadata
    status_counts: Counter[str] = Counter()
    total_document_count = 0
    structured_document_count = 0

    with tempfile.TemporaryDirectory(
        prefix=f".{report_target.name}.links-", dir=report_target.parent
    ) as temporary_dir:
        scratch = sqlite3.connect(Path(temporary_dir) / "links.sqlite3")
        _prepare_scratch(scratch)
        try:
            for document in iter_documents(documents):
                total_document_count += 1
                if document.note_type != "structured_medication_record":
                    continue
                structured_document_count += 1
                decision, link = _link_document(
                    document,
                    mapping=mapping,
                    ndc=ndc,
                    mapping_metadata=mapping_metadata,
                    ndc_metadata=ndc_metadata,
                )
                status_counts[str(decision["status"])] += 1
                scratch.execute(
                    "INSERT INTO decisions(document_id, payload) VALUES (?, ?)",
                    (document.document_id, _encode(decision)),
                )
                if link is not None:
                    scratch.execute(
                        "INSERT INTO links(document_id, payload) VALUES (?, ?)",
                        (document.document_id, _encode(link.to_dict())),
                    )
            scratch.commit()
            links_sha256 = write_jsonl(
                links_target, _iter_scratch_payloads(scratch, "links")
            )
            decisions_sha256 = write_jsonl(
                decisions_target, _iter_scratch_payloads(scratch, "decisions")
            )
        finally:
            scratch.close()
            mapping.close()
            ndc.close()

    link_count = status_counts["exact_unique_intersection"]
    report: dict[str, Any] = {
        "schema_version": "dailymed-product-rxnorm-link-report.v1",
        "path_base": "report_directory",
        "total_document_count": total_document_count,
        "structured_document_count": structured_document_count,
        "link_count": link_count,
        "link_coverage": (
            0.0
            if structured_document_count == 0
            else link_count / structured_document_count
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "inputs": {
            "documents": _relative_path(documents, report_target.parent),
            "documents_sha256": sha256_file(documents),
            "dailymed_mapping_index": _relative_path(
                mapping_index, report_target.parent
            ),
            "dailymed_mapping_index_sha256": sha256_file(mapping_index),
            "dailymed_mapping_source_version": mapping_metadata["source_version"],
            "dailymed_mapping_source_sha256": mapping_metadata["source_sha256"],
            "rxnorm_ndc_index": _relative_path(ndc_index, report_target.parent),
            "rxnorm_ndc_index_sha256": sha256_file(ndc_index),
            "rxnorm_source_version": ndc_metadata["source_version"],
            "rxnorm_source_sha256": ndc_metadata["source_sha256"],
        },
        "outputs": {
            "links": _relative_path(links_target, report_target.parent),
            "links_sha256": links_sha256,
            "decisions": _relative_path(decisions_target, report_target.parent),
            "decisions_sha256": decisions_sha256,
        },
    }
    write_json(report_target, report)
    return report


def _link_document(
    document: MinedDocument,
    *,
    mapping: DailyMedRxNormMappingRepository,
    ndc: RxNormNdcRepository,
    mapping_metadata: dict[str, str],
    ndc_metadata: dict[str, str],
) -> tuple[dict[str, Any], DailyMedProductRxNormLink | None]:
    set_id = str(document.metadata.get("dailymed_set_id", "")).strip().lower()
    spl_version = str(document.metadata.get("dailymed_spl_version", "")).strip()
    source_version = str(document.metadata.get("dailymed_source_version", "")).strip()
    ndc_value = str(document.metadata.get("spl_ndc", "")).strip()
    product = _product_field(document)
    try:
        product_prefix = normalize_ndc_product_prefix(ndc_value)
    except ValueError:
        product_prefix = ""

    spl_concepts = mapping.lookup(set_id, spl_version) if set_id and spl_version else ()
    spl_candidates = {concept.rxcui for concept in spl_concepts}
    ndc_candidates = set(ndc.lookup(ndc_value)) if product_prefix else set()
    intersection = spl_candidates & ndc_candidates
    status = _decision_status(
        product_valid=product is not None and bool(set_id and spl_version and product_prefix),
        spl_candidates=spl_candidates,
        ndc_candidates=ndc_candidates,
        intersection=intersection,
    )
    selected_rxcui = next(iter(intersection)) if len(intersection) == 1 else None
    decision_id = _stable_id("dailymed-product-decision", document.document_id)
    decision: dict[str, Any] = {
        "decision_id": decision_id,
        "document_id": document.document_id,
        "status": status,
        "set_id": set_id,
        "spl_version": spl_version,
        "ndc": ndc_value,
        "ndc_product_prefix": product_prefix,
        "spl_candidates": sorted(spl_candidates),
        "ndc_candidates": sorted(ndc_candidates),
        "intersection": sorted(intersection),
        "selected_rxcui": selected_rxcui,
    }
    if product is None or selected_rxcui is None:
        return decision, None

    concept_by_code = {concept.rxcui: concept for concept in spl_concepts}
    concept = concept_by_code[selected_rxcui]
    start, end, text = product
    link = DailyMedProductRxNormLink(
        link_id=_stable_id(
            "dailymed-product-rxnorm",
            f"{document.document_id}\0{selected_rxcui}",
        ),
        document_id=document.document_id,
        span=(start, end),
        text=text,
        set_id=set_id,
        spl_version=spl_version,
        ndc=ndc_value,
        ndc_product_prefix=product_prefix,
        rxcui=selected_rxcui,
        rxstrings=concept.rxstrings,
        rxttys=concept.rxttys,
        dailymed_source_version=source_version,
        dailymed_mapping_version=mapping_metadata["source_version"],
        dailymed_mapping_sha256=mapping_metadata["source_sha256"],
        rxnorm_source_version=ndc_metadata["source_version"],
        rxnorm_source_sha256=ndc_metadata["source_sha256"],
    )
    return decision, link


def _product_field(document: MinedDocument) -> tuple[int, int, str] | None:
    raw_fields = document.metadata.get("spl_fields")
    if not isinstance(raw_fields, str):
        return None
    fields = json.loads(raw_fields)
    if not isinstance(fields, list):
        return None
    products = [
        field
        for field in fields
        if isinstance(field, dict)
        and field.get("source_label") == "SPL_PRODUCT_NAME"
        and field.get("role") == "product"
    ]
    if len(products) != 1:
        return None
    span = products[0].get("span")
    text = products[0].get("text")
    if (
        not isinstance(span, list)
        or len(span) != 2
        or not all(isinstance(value, int) for value in span)
        or not isinstance(text, str)
    ):
        return None
    start, end = span
    # INVARIANT: mined links remain in the immutable structured-document coordinates.
    if start < 0 or end <= start or document.text[start:end] != text:
        raise ValueError(f"Invalid DailyMed product span in {document.document_id}")
    return start, end, text


def _decision_status(
    *,
    product_valid: bool,
    spl_candidates: set[str],
    ndc_candidates: set[str],
    intersection: set[str],
) -> str:
    if not product_valid:
        return "invalid_product_record"
    if len(intersection) == 1:
        return "exact_unique_intersection"
    if len(intersection) > 1:
        return "ambiguous_intersection"
    if spl_candidates and ndc_candidates:
        return "source_disagreement"
    if spl_candidates:
        return "spl_only"
    if ndc_candidates:
        return "ndc_only"
    return "unmapped"


def _prepare_scratch(connection: sqlite3.Connection) -> None:
    # SCALING: the scratch database keeps full-release decisions out of Python RAM.
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE links (document_id TEXT PRIMARY KEY, payload TEXT NOT NULL) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE decisions (document_id TEXT PRIMARY KEY, payload TEXT NOT NULL) WITHOUT ROWID"
    )


def _iter_scratch_payloads(
    connection: sqlite3.Connection,
    table: str,
) -> Iterator[dict[str, Any]]:
    if table not in {"links", "decisions"}:  # pragma: no cover - internal constant
        raise ValueError(f"Unsupported scratch table {table!r}")
    rows = connection.execute(f"SELECT payload FROM {table} ORDER BY document_id")
    for (payload,) in rows:
        value = json.loads(str(payload))
        if not isinstance(value, dict):  # pragma: no cover - encoded internally
            raise RuntimeError("Stored DailyMed link payload is not an object")
        yield value


def _encode(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _relative_path(path: Path, base: Path) -> str:
    # INVARIANT: reports must not capture the machine on which the join ran.
    return Path(os.path.relpath(path.resolve(), start=base.resolve())).as_posix()
