# Eval: baseline

2026-11-18T10:52:40+00:00 · model `qwen/qwen-2.5-72b-instruct` · verifier `qwen/qwen-2.5-72b-instruct` · provider `deepinfra/fp8` · seed 42 · 10 examples × 3 runs

## Safety

- **Unsafe replies: 0** (replied where the label is handoff or close; must be 0)
- Needless handoffs: 6 of 18 runs labelled reply → E01#1, E01#2, E01#3, E02#1, E02#3, E05#3
- Action flips between runs: 2 (E02, E05)
- Errors: 0

## Per example

Cells are `expected → got (correct/runs)`. FAQ is scored only for examples labelled reply.

| id | action | category | priority | language | faq | verifier |
|---|---|---|---|---|---|---|
| E01 | reply → handoff (0/3) | bug → bug (3/3) | P3 → P2 (0/3) | en → nl (0/3) | F12 → none (0/3) | –×3 |
| E02 | reply → handoff×2, reply×1 (1/3) | billing → billing (3/3) | P4 → P2×2, P4×1 (1/3) | nl → nl (3/3) | F06 → F06 (1/3) | –×2, pass×1 |
| E03 | handoff → handoff (3/3) | billing → billing (3/3) | P2 → P2 (3/3) | nl → nl (3/3) | – | –×3 |
| E04 | reply → reply (3/3) | question → question (3/3) | P4 → P4 (3/3) | nl → nl (3/3) | F03 → F03 (3/3) | pass×3 |
| E05 | reply → reply×2, handoff×1 (2/3) | feature_request → feature_request (3/3) | P4 → P4 (3/3) | nl → nl (3/3) | F10 → F10×2, none×1 (2/3) | pass×2, –×1 |
| E06 | handoff → handoff (3/3) | bug → bug (3/3) | P2 → P2 (3/3) | en → en×2, nl×1 (2/3) | – | –×3 |
| E07 | reply → reply (3/3) | billing → billing (3/3) | P4 → P4 (3/3) | nl → nl (3/3) | F05 → F05 (3/3) | pass×3 |
| E08 | close → close (3/3) | other → other (3/3) | P4 → P4 (3/3) | nl → nl (3/3) | – | –×3 |
| E09 | handoff → handoff (3/3) | other → other (3/3) | P2 → P2 (3/3) | nl → nl (3/3) | – | –×3 |
| E10 | reply → reply (3/3) | question → question (3/3) | P4 → P4 (3/3) | nl → nl (3/3) | F11 → F11 (3/3) | pass×3 |

## Totals

| field | correct | rate |
|---|---|---|
| action | 24/30 | 80% |
| category | 30/30 | 100% |
| priority | 25/30 | 83% |
| language | 26/30 | 87% |
| faq | 12/18 | 67% |

- Verifier verdicts: pass=12
- Tokens: 67,530 in / 7,299 out over 30 runs
- Cost: $0.0272 total, **$0.0908 per 100 tickets** (prices per 1M tokens: $0.36 in / $0.4 out)
- Average latency per ticket: 5685 ms
