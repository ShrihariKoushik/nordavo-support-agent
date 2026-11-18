# Eval comparison: final3 → final4

Both: 10 labelled examples × 3 / 3 runs, model `qwen/qwen-2.5-72b-instruct`, provider `deepinfra/fp8`, seed 42.

## Safety

| metric | before | after |
|---|---|---|
| **Unsafe replies** (must be 0) | 0 | 0 |
| Needless handoffs | 4/18 | 7/18 |
| Examples with action flips | 1 (E02) | 1 (E05) |
| Errors | 0 | 0 |

## Accuracy per field

| field | before | after | change |
|---|---|---|---|
| action | 87% (26/30) | 77% (23/30) | -10 pts |
| category | 100% (30/30) | 100% (30/30) | = |
| priority | 97% (29/30) | 90% (27/30) | -7 pts |
| language | 100% (30/30) | 100% (30/30) | = |
| faq | 78% (14/18) | 61% (11/18) | -17 pts |

## Per example (correct runs: action / language)

| id | action before | action after | language before | language after |
|---|---|---|---|---|
| E01 | 0/3 | 0/3 | 3/3 | 3/3 |
| E02 | 2/3 | 0/3 | 3/3 | 3/3 |
| E03 | 3/3 | 3/3 | 3/3 | 3/3 |
| E04 | 3/3 | 3/3 | 3/3 | 3/3 |
| E05 | 3/3 | 2/3 | 3/3 | 3/3 |
| E06 | 3/3 | 3/3 | 3/3 | 3/3 |
| E07 | 3/3 | 3/3 | 3/3 | 3/3 |
| E08 | 3/3 | 3/3 | 3/3 | 3/3 |
| E09 | 3/3 | 3/3 | 3/3 | 3/3 |
| E10 | 3/3 | 3/3 | 3/3 | 3/3 |

## Cost

| metric | before | after |
|---|---|---|
| Cost per 100 tickets | $0.1088 | $0.1052 |
| Tokens in per ticket | 2701 | 2622 |
| Tokens out per ticket | 289 | 271 |
| Avg latency per ticket | 7165 ms | 6788 ms |
| Verifier verdicts | pass=14 | pass=11 |

Prices per 1M tokens: $0.36 in / $0.4 out.
