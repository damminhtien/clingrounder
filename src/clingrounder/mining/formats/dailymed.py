"""Structured DailyMed SPL fields rendered into offset-safe medication records."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from clingrounder.schema.types import EntityType

__all__ = [
    "SplCodedTerm",
    "SplIngredientRecord",
    "SplProductRecord",
    "SplRenderedField",
    "SplRenderedProduct",
    "extract_spl_products",
    "render_spl_product",
]

_NDC_CODE_SYSTEM_OID = "2.16.840.1.113883.6.69"
_UNII_CODE_SYSTEM_OID = "2.16.840.1.113883.4.9"


@dataclass(frozen=True)
class SplCodedTerm:
    """One display term and its source terminology code, when present."""

    text: str
    code: str = ""
    code_system: str = ""


@dataclass(frozen=True)
class SplIngredientRecord:
    """One active ingredient and its structured SPL strength."""

    ingredient: SplCodedTerm
    strength: str = ""


@dataclass(frozen=True)
class SplProductRecord:
    """Normalized fields extracted from one manufactured product node."""

    product_name: str
    ndc: str
    generic_names: tuple[str, ...]
    active_ingredients: tuple[SplIngredientRecord, ...]
    dosage_form: SplCodedTerm | None
    routes: tuple[SplCodedTerm, ...]


@dataclass(frozen=True)
class SplRenderedField:
    """One source-derived field projected onto the rendered record text."""

    span: tuple[int, int]
    text: str
    entity_type: str
    source_label: str
    role: str
    group_id: str
    code_system: str = ""
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "span": [self.span[0], self.span[1]],
            "text": self.text,
            "entity_type": self.entity_type,
            "source_label": self.source_label,
            "role": self.role,
            "group_id": self.group_id,
            "code_system": self.code_system,
            "code": self.code,
        }


@dataclass(frozen=True)
class SplRenderedProduct:
    """Deterministic product text plus exact source-field offsets."""

    text: str
    fields: tuple[SplRenderedField, ...]


def extract_spl_products(root: ET.Element) -> tuple[SplProductRecord, ...]:
    """Extract actual products while ignoring outer SPL relationship wrappers."""

    candidates: list[tuple[ET.Element, ET.Element]] = []
    seen_nodes: set[int] = set()
    for wrapper in _all_local(root, "manufacturedProduct"):
        nested_products = _direct_children(wrapper, "manufacturedProduct")
        for product in nested_products:
            if id(product) not in seen_nodes and _looks_like_product(product):
                candidates.append((product, wrapper))
                seen_nodes.add(id(product))
        if id(wrapper) not in seen_nodes and _looks_like_product(wrapper):
            candidates.append((wrapper, wrapper))
            seen_nodes.add(id(wrapper))

    records: list[SplProductRecord] = []
    seen_records: set[SplProductRecord] = set()
    for product, context in candidates:
        record = _extract_product(product, context)
        if record is None or record in seen_records:
            continue
        records.append(record)
        seen_records.add(record)
    return tuple(records)


def render_spl_product(record: SplProductRecord) -> SplRenderedProduct:
    """Render structured fields and compute offsets in one append-only pass."""

    lines: list[str] = []
    fields: list[SplRenderedField] = []

    def append_field(
        label: str,
        value: str,
        *,
        entity_type: EntityType,
        source_label: str,
        role: str,
        group_id: str,
        code_system: str = "",
        code: str = "",
    ) -> None:
        prefix = "\n".join(lines)
        line_start = len(prefix) + (1 if lines else 0)
        value_start = line_start + len(label) + 2
        lines.append(f"{label}: {value}")
        fields.append(
            SplRenderedField(
                span=(value_start, value_start + len(value)),
                text=value,
                entity_type=entity_type.value,
                source_label=source_label,
                role=role,
                group_id=group_id,
                code_system=code_system,
                code=code,
            )
        )

    append_field(
        "Product",
        record.product_name,
        entity_type=EntityType.DRUG,
        source_label="SPL_PRODUCT_NAME",
        role="product",
        group_id="product",
        code_system="NDC" if record.ndc else "",
        code=record.ndc,
    )
    for index, generic_name in enumerate(record.generic_names):
        append_field(
            "Generic name",
            generic_name,
            entity_type=EntityType.DRUG,
            source_label="SPL_GENERIC_NAME",
            role="generic_name",
            group_id=f"generic:{index}",
        )
    for index, ingredient in enumerate(record.active_ingredients):
        group_id = f"ingredient:{index}"
        append_field(
            "Active ingredient",
            ingredient.ingredient.text,
            entity_type=EntityType.DRUG,
            source_label="SPL_ACTIVE_INGREDIENT",
            role="active_ingredient",
            group_id=group_id,
            code_system="UNII" if ingredient.ingredient.code else "",
            code=ingredient.ingredient.code,
        )
        if ingredient.strength:
            append_field(
                "Strength",
                ingredient.strength,
                entity_type=EntityType.STRENGTH,
                source_label="SPL_INGREDIENT_STRENGTH",
                role="strength",
                group_id=group_id,
            )
    if record.dosage_form is not None:
        append_field(
            "Dosage form",
            record.dosage_form.text,
            entity_type=EntityType.DOSAGE_FORM,
            source_label="SPL_DOSAGE_FORM",
            role="dosage_form",
            group_id="product",
            code_system="NCI_THESAURUS" if record.dosage_form.code else "",
            code=record.dosage_form.code,
        )
    for index, route in enumerate(record.routes):
        append_field(
            "Route",
            route.text,
            entity_type=EntityType.ROUTE,
            source_label="SPL_ROUTE",
            role="route",
            group_id=f"route:{index}",
            code_system="NCI_THESAURUS" if route.code else "",
            code=route.code,
        )
    if record.ndc:
        lines.append(f"NDC: {record.ndc}")

    text = "\n".join(lines)
    # INVARIANT: source fields are projected only after final text assembly.
    for field in fields:
        start, end = field.span
        if text[start:end] != field.text:
            raise AssertionError(f"DailyMed rendered offset mismatch for {field.text!r}")
    return SplRenderedProduct(text=text, fields=tuple(fields))


def _extract_product(
    product: ET.Element,
    context: ET.Element,
) -> SplProductRecord | None:
    name_node = _direct_child(product, "name")
    product_name = "" if name_node is None else _node_text(name_node)
    if not product_name:
        return None
    code_node = _direct_child(product, "code")
    ndc = ""
    if code_node is not None and code_node.get("codeSystem") == _NDC_CODE_SYSTEM_OID:
        ndc = (code_node.get("code") or "").strip()
    generic_names = _unique_strings(
        _node_text(name)
        for generic in _direct_children(product, "asEntityWithGeneric")
        for medicine in _direct_children(generic, "genericMedicine")
        for name in _direct_children(medicine, "name")
    )
    ingredients = tuple(
        ingredient
        for node in _direct_children(product, "ingredient")
        if (node.get("classCode") or "").upper() == "ACTIB"
        if (ingredient := _extract_active_ingredient(node)) is not None
    )
    form_node = _direct_child(product, "formCode")
    dosage_form = None if form_node is None else _coded_term(form_node)
    routes = _unique_terms(
        _coded_term(node)
        for node in context.iter()
        if _local_name(node.tag) == "routeCode"
    )
    return SplProductRecord(
        product_name=product_name,
        ndc=ndc,
        generic_names=generic_names,
        active_ingredients=ingredients,
        dosage_form=dosage_form,
        routes=routes,
    )


def _extract_active_ingredient(node: ET.Element) -> SplIngredientRecord | None:
    substance = _direct_child(node, "ingredientSubstance")
    if substance is None:
        return None
    name = _direct_child(substance, "name")
    ingredient_name = "" if name is None else _node_text(name)
    if not ingredient_name:
        return None
    code_node = _direct_child(substance, "code")
    code = ""
    code_system = ""
    if code_node is not None:
        code = (code_node.get("code") or "").strip()
        code_system = (code_node.get("codeSystem") or "").strip()
    if code_system and code_system != _UNII_CODE_SYSTEM_OID:
        code = ""
        code_system = ""
    quantity = _direct_child(node, "quantity")
    return SplIngredientRecord(
        ingredient=SplCodedTerm(
            text=ingredient_name,
            code=code,
            code_system=code_system,
        ),
        strength="" if quantity is None else _format_quantity(quantity),
    )


def _format_quantity(quantity: ET.Element) -> str:
    numerator = _direct_child(quantity, "numerator")
    denominator = _direct_child(quantity, "denominator")
    numerator_text = _quantity_part(numerator)
    denominator_text = _quantity_part(denominator)
    if not numerator_text:
        return ""
    if not denominator_text or denominator_text == "1":
        return numerator_text
    return f"{numerator_text} per {denominator_text}"


def _quantity_part(node: ET.Element | None) -> str:
    if node is None:
        return ""
    value = (node.get("value") or "").strip()
    unit = (node.get("unit") or "").strip()
    return " ".join(part for part in (value, unit) if part)


def _coded_term(node: ET.Element) -> SplCodedTerm:
    return SplCodedTerm(
        text=(node.get("displayName") or _node_text(node)).strip(),
        code=(node.get("code") or "").strip(),
        code_system=(node.get("codeSystem") or "").strip(),
    )


def _looks_like_product(node: ET.Element) -> bool:
    return _direct_child(node, "name") is not None and any(
        _direct_child(node, local_name) is not None
        for local_name in ("code", "formCode", "ingredient")
    )


def _direct_child(node: ET.Element, local_name: str) -> ET.Element | None:
    return next(
        (child for child in node if _local_name(child.tag) == local_name),
        None,
    )


def _direct_children(node: ET.Element, local_name: str) -> tuple[ET.Element, ...]:
    return tuple(child for child in node if _local_name(child.tag) == local_name)


def _all_local(root: ET.Element, local_name: str) -> tuple[ET.Element, ...]:
    return tuple(node for node in root.iter() if _local_name(node.tag) == local_name)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _node_text(node: ET.Element) -> str:
    return " ".join("".join(node.itertext()).split())


def _unique_strings(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _unique_terms(values: Any) -> tuple[SplCodedTerm, ...]:
    result: list[SplCodedTerm] = []
    for value in values:
        if value.text and value not in result:
            result.append(value)
    return tuple(result)
