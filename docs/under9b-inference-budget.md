# Under-9B Inference Budget

The maximum-score Phase 1 pipeline uses one learned artifact multiple times but counts its
parameters once. Quantization changes memory usage, not this budget.

Verify the pinned plan with:

```bash
uv run medical-kg model inspect-inference-budget \
  --config configs/benchmarks/phase1/models/phase1-under9b-max.yaml \
  --output outputs/models/under9b-budget.json
```

The verifier distinguishes:

- `active`: an existing artifact whose count is read from a pinned training manifest or directly
  from a Safetensors header;
- `reserved`: a hard upper bound for an untrained artifact. A reservation has no fake model
  revision and cannot be loaded for inference.

The checked-in plan uses a limit of `8,900,000,000`, not the competition boundary of 9B. Any
changed manifest, Safetensors file, declared count, or reservation that crosses this limit fails
before model loading. When a reserved component is trained, replace its reservation with an active
entry and attach measured evidence.
