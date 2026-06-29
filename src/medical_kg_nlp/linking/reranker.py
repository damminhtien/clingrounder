from __future__ import annotations
from medical_kg_nlp.linking.candidate import Candidate


class HeuristicReranker:
    def rerank(self, candidates: list[Candidate], context_window: str = "") -> list[Candidate]:
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)

