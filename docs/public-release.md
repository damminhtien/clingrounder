# Public Release Policy

The public repository contains the reusable clinical NLP toolkit, synthetic fixtures, source
policies, processing dossiers, and content-addressed manifests. It does not publish restricted
clinical text, competition labels, licensed archives, generated predictions, or model weights.

This is a storage boundary, not a deletion policy:

```text
Git repository       code + configs + fixtures + provenance + checksums
local/CAS storage    raw corpora + licensed terminology + annotations + checkpoints + runs
```

## Two Complementary Contracts

`medical-kg data release lock` fingerprints every input needed to reproduce one mining or model
release. It can reference external content-addressed objects that are restored on another machine.

`medical-kg release audit` inspects the Git index before publication. It fails when:

- a protected path has no explicit policy rule;
- `local_only` or `manifest_only` bytes are tracked;
- a required provenance file or attribution notice is absent;
- an unapproved tracked file exceeds the repository size limit;
- a tracked text file contains a recognized credential shape.

The audit never prints credential contents. Rules are evaluated in YAML order and the selected
`rule_id` is included in every finding, so publication behavior is inspectable rather than hidden
in shell scripts.

```bash
uv run medical-kg release audit \
  --policy configs/repository/public-release.yaml \
  --root .
```

Before removing restricted bytes from the Git index, or after refreshing an authorized local
snapshot, rebuild its deterministic inventory:

```bash
uv run medical-kg release inventory \
  --policy configs/repository/public-release.yaml \
  --root . \
  --output data/provenance/local-artifacts.json
```

Only rules with `inventory_local_bytes: true` are traversed. This keeps canonical terminology and
reviewed supervision verifiable while excluding transient run directories from the stable manifest.

## Reproducing Restricted Experiments

1. Clone the public repository and install the locked environment.
2. Read `data/sources/mining_registry.yaml` for access, license, retention, and permitted-use rules.
3. Read `data/sources/processing_status.yaml` for the exact connector, config, dossier, and artifact
   roots used by each source.
4. Acquire authorized bytes into local or S3 content-addressed storage.
5. Run the documented rebuild steps and verify the archived release lock.

Absence from Git does not weaken provenance: SHA-256 identities, source versions, parser versions,
configs, and rebuild commands remain public. A source whose terms prohibit redistribution can
therefore be reproduced by an authorized user without exposing its bytes to everyone else.

## Adding Data

Every new tracked file under `data/`, `outputs/`, `models/`, or `checkpoints/` needs a matching rule
in `configs/repository/public-release.yaml`. Prefer a tiny synthetic fixture for tests. Put real
corpora and generated artifacts in external storage and publish a release lock or import manifest.
