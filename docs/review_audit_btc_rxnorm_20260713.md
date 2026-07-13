# BTC RxNorm Public Probe - 2026-07-13

## Promoted Baseline

- Public score: `43.2014` (`+1.3437`)
- WER: `50.8167` (unchanged)
- J assertion: `49.7245` (unchanged)
- J candidates: `33.8226` (`+3.3592`)
- Artifact: `outputs/phase1/20260713_btc_rxnorm_isolated/output.zip`
- SHA-256: `73321a5332bc15523d49d6df073ea4ea089d4106b2be564766fa30faf1bf4633`

The artifact froze the public-winning entity and assertion output and changed only RxNorm
candidates on 176 exact-aligned medication entities. The public candidate isolation gate passed,
so this replaces `A_NEG_HIST` (`41.8577`) as the campaign baseline. ICD candidate probes remain
blocked because `C_ICD20` independently reduced J candidates.

## Next Probe

The promoted artifact emits 354 codes on 176 medications: 86 rows have one candidate and 90 rows
have between two and five. A top-1-only ablation leaves entities and assertions unchanged and
improves local manual-gold score from `62.3285` to `62.8152`; holdout improves from `62.2807` to
`62.8000`. The next isolated public probe is therefore `C_BTC_RXNORM_TOP1`.

Do not add fuzzy retrieval, ICD mappings, family assertions, or entity changes to that probe. After
top-1, split product-aware SCD selection from ingredient fallback as separate candidate-only
experiments.
