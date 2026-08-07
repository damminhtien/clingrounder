"""Strict recovery of JSON values emitted by local generative models."""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = ["StructuredResponseError", "parse_structured_response"]

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE)


class StructuredResponseError(ValueError):
    """Raised when a model response contains no complete JSON object or array."""


def parse_structured_response(raw_response: str) -> Any:
    """Parse the first complete JSON value after removing known presentation wrappers.

    The decoder still performs all syntax validation. Scanning only locates a possible JSON start;
    it does not repair malformed fields or accept Python literals.
    """

    if not raw_response.strip():
        raise StructuredResponseError("Model response is empty")
    without_thinking = _THINK_BLOCK.sub("", raw_response).strip()
    fenced = _FENCE.search(without_thinking)
    candidate = fenced.group(1).strip() if fenced else without_thinking
    decoder = json.JSONDecoder()
    errors: list[str] = []
    for index, character in enumerate(candidate):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError as error:
            errors.append(str(error))
            continue
        if not isinstance(value, (dict, list)):
            raise StructuredResponseError("Structured response must be a JSON object or array")
        return value
    detail = errors[-1] if errors else "no JSON object or array start found"
    raise StructuredResponseError(f"Could not parse structured model response: {detail}")
