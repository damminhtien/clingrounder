# Security And Data Governance

This repository implements practical local-runtime controls. It does **not**
claim HIPAA, GDPR, ISO 27001, SOC 2, or any other regulatory certification.
Deployment owners remain responsible for access control, contracts, incident
response, retention approval, and legal review.

## Threat model

| Threat | Control in this repository | Residual risk / owner action |
| --- | --- | --- |
| PHI in logs or traces | hashed document IDs, no raw text in audit events, default warning logging | review custom sinks and host logs |
| malicious document content | bounded parsing, immutable raw offsets, no document execution | sandbox untrusted ingestion |
| oversized input / denial of service | profile limits, bounded trace retention, batch limits | enforce service-level quotas |
| dependency compromise | lockfile, Dependabot, `pip-audit`, SBOM artifact | review advisories and pin trusted indexes |
| model supply chain | pinned model revision, local-only loading, SHA/allowlist APIs | provide and review model manifest hashes |
| terminology tampering | source/index fingerprints and SHA verification APIs | protect the storage volume and signing keys |
| path traversal | resolved allowed-root checks and relative model subfolders | configure roots narrowly |
| unsafe local model loading | explicit local-file policy; no implicit hosted fallback | review custom adapters and pickle formats |
| trace / log leakage | bounded PHI-safe audit schema and redacted errors | configure external exporters carefully |
| temporary-file attacks | atomic build patterns and exclusive temporary paths | use encrypted volumes for sensitive data |

## Data policy

Use a profile-owned governance block when a deployment needs explicit policy:

```yaml
governance:
  local_files_only: true
  allowed_artifact_roots: [./data, ./.cache/models]
  artifact_allowlist:
    - path: ./data/dictionaries/seed_concepts.jsonl
      sha256: <64-lowercase-hex-digest>
  data:
    logging_level: WARNING
    text_retention: none
    trace_retention: memory_only
    hash_document_ids: true
    metadata_allowlist: [language, source]
    deletion_behavior: best_effort_unlink
```

`verify_artifact()` fails closed on a digest mismatch. Model and terminology
release manifests should be stored with the deployment, not inferred from
untrusted document content. The model adapters already use local-only loading;
the governance API provides the missing hash and allowlist enforcement point.

## Audit events

The runtime emits bounded events for profile load, terminology/model load,
prediction, and validation failure. Events include fingerprints and outcome
metadata, never note text or mention text. Administrative configuration changes
should be emitted by the deployment/configuration layer because the library
cannot observe changes made outside the process.

## CI artifacts

CI produces dependency vulnerability and SBOM JSON artifacts. These are review
inputs, not proof that a deployment is secure. Secret scanning is also run on
the checked-out repository; credentials must still be revoked if exposed.

Not implemented here: key management, artifact signing service, network policy,
identity/access management, encrypted storage, breach response, or formal
compliance certification.
