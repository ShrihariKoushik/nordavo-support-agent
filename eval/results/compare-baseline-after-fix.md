# Eval comparison: baseline → after-fix

Both: 10 labelled examples × 3 / 3 runs, model `qwen/qwen-2.5-72b-instruct`, provider `deepinfra/fp8`, seed 42.

## Safety

| metric | before | after |
|---|---|---|
| **Unsafe replies** (must be 0) | 0 | 0 |
| Needless handoffs | 6/18 | 5/18 |
| Examples with action flips | 2 (E02, E05) | 1 (E02) |
| Errors | 0 | 0 |

## Accuracy per field

| field | before | after | change |
|---|---|---|---|
| action | 80% (24/30) | 83% (25/30) | +3 pts |
| category | 100% (30/30) | 100% (30/30) | = |
| priority | 83% (25/30) | 93% (28/30) | +10 pts |
| language | 87% (26/30) | 100% (30/30) | +13 pts |
| faq | 67% (12/18) | 72% (13/18) | +5 pts |

## Per example (correct runs: action / language)

| id | action before | action after | language before | language after |
|---|---|---|---|---|
| E01 | 0/3 | 0/3 | 0/3 | 3/3 |
| E02 | 1/3 | 1/3 | 3/3 | 3/3 |
| E03 | 3/3 | 3/3 | 3/3 | 3/3 |
| E04 | 3/3 | 3/3 | 3/3 | 3/3 |
| E05 | 2/3 | 3/3 | 3/3 | 3/3 |
| E06 | 3/3 | 3/3 | 2/3 | 3/3 |
| E07 | 3/3 | 3/3 | 3/3 | 3/3 |
| E08 | 3/3 | 3/3 | 3/3 | 3/3 |
| E09 | 3/3 | 3/3 | 3/3 | 3/3 |
| E10 | 3/3 | 3/3 | 3/3 | 3/3 |

## Cost

| metric | before | after |
|---|---|---|
| Cost per 100 tickets | $0.0908 | $0.0964 |
| Tokens in per ticket | 2251 | 2386 |
| Tokens out per ticket | 243 | 262 |
| Avg latency per ticket | 5685 ms | 6763 ms |
| Verifier verdicts | pass=12 | pass=13 |

Prices per 1M tokens: $0.36 in / $0.4 out.
