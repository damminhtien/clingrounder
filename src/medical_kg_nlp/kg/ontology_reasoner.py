from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from medical_kg_nlp.dictionaries.dictionary_store import DictionaryStore
from medical_kg_nlp.schema.types import CodeSystem


OntologyNode = tuple[CodeSystem, str]


@dataclass(frozen=True)
class ReasoningPath:
    child: OntologyNode
    ancestor: OntologyNode
    path: tuple[OntologyNode, ...]


class OntologyReasoner:
    """Dictionary-backed hierarchy closure with explicit path provenance."""

    def __init__(self, store: DictionaryStore) -> None:
        self.parents: dict[OntologyNode, set[OntologyNode]] = {}
        nodes_by_concept = {
            entry.concept_id: (entry.code_system, entry.code)
            for entry in store.entries
            if entry.code is not None
        }
        nodes_by_code = {
            (entry.code_system, entry.code): (entry.code_system, entry.code)
            for entry in store.entries
            if entry.code is not None
        }
        for entry in store.entries:
            if entry.code is None:
                continue
            child = (entry.code_system, entry.code)
            self.parents.setdefault(child, set())
            for parent_value in entry.parents:
                parent = nodes_by_concept.get(parent_value) or nodes_by_code.get(
                    (entry.code_system, parent_value)
                )
                if parent is not None and parent != child:
                    self.parents[child].add(parent)
                    self.parents.setdefault(parent, set())

    def contains(self, code_system: CodeSystem, code: str) -> bool:
        return (code_system, code) in self.parents

    def ancestors(self, code_system: CodeSystem, code: str) -> dict[OntologyNode, int]:
        start = (code_system, code)
        distances: dict[OntologyNode, int] = {}
        queue: deque[tuple[OntologyNode, int]] = deque(
            (parent, 1) for parent in self.parents.get(start, ())
        )
        while queue:
            node, distance = queue.popleft()
            previous = distances.get(node)
            if previous is not None and previous <= distance:
                continue
            distances[node] = distance
            queue.extend((parent, distance + 1) for parent in self.parents.get(node, ()))
        return distances

    def is_a(
        self,
        code_system: CodeSystem,
        child_code: str,
        ancestor_code: str,
    ) -> bool:
        if child_code == ancestor_code:
            return True
        return (code_system, ancestor_code) in self.ancestors(code_system, child_code)

    def hierarchy_distance(
        self,
        code_system: CodeSystem,
        child_code: str,
        ancestor_code: str,
    ) -> int | None:
        if child_code == ancestor_code:
            return 0
        return self.ancestors(code_system, child_code).get((code_system, ancestor_code))

    def explain_is_a(
        self,
        code_system: CodeSystem,
        child_code: str,
        ancestor_code: str,
    ) -> ReasoningPath | None:
        start = (code_system, child_code)
        target = (code_system, ancestor_code)
        if start == target:
            return ReasoningPath(child=start, ancestor=target, path=(start,))
        queue: deque[OntologyNode] = deque([start])
        previous: dict[OntologyNode, OntologyNode | None] = {start: None}
        while queue:
            node = queue.popleft()
            for parent in sorted(
                self.parents.get(node, ()), key=lambda item: (item[0].value, item[1])
            ):
                if parent in previous:
                    continue
                previous[parent] = node
                if parent == target:
                    path: list[OntologyNode] = [target]
                    current: OntologyNode | None = node
                    while current is not None:
                        path.append(current)
                        current = previous[current]
                    path.reverse()
                    return ReasoningPath(child=start, ancestor=target, path=tuple(path))
                queue.append(parent)
        return None
