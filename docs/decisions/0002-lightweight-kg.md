# 0002: Lightweight KG First

## Status

Accepted.

## Context

The current graph needs are code-system compatibility, relation-type validation, and simple ontology
edges. A graph database would add operational cost before graph traversal requirements are clear.

## Decision

Represent KG data with local tables, dictionaries, and small in-memory structures first. Defer Neo4j
or another graph database until interactive traversal or graph-scale querying is required.

## Consequences

- Lower setup cost for experiments and CI.
- KG constraints remain easy to unit test.
- Future graph storage must preserve the current validator behavior.
