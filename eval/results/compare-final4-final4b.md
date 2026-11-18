# Eval comparison: final4 → final4b

Both: 10 labelled examples × 3 / 3 runs, model `qwen/qwen-2.5-72b-instruct`, provider `deepinfra/fp8`, seed 42.

## Safety

| metric | before | after |
|---|---|---|
| **Unsafe replies** (must be 0) | 0 | 0 |
| Needless handoffs | 7/18 | 6/18 |
| Examples with action flips | 1 (E05) | 0 |
| Errors | 0 | 0 |

## Accuracy per field

| field | before | after | change |
|---|---|---|---|
| action | 77% (23/30) | 80% (24/30) | +3 pts |
| category | 100% (30/30) | 100% (30/30) | = |
| priority | 90% (27/30) | 90% (27/30) | = |
| language | 100% (30/30) | 100% (30/30) | = |
| faq | 61% (11/18) | 67% (12/18) | +6 pts |

## Per example (correct runs: action / language)

| id | action before | action after | language before | language after |
|---|---|---|---|---|
| E01 | 0/3 | 0/3 | 3/3 | 3/3 |
| E02 | 0/3 | 0/3 | 3/3 | 3/3 |
| E03 | 3/3 | 3/3 | 3/3 | 3/3 |
| E04 | 3/3 | 3/3 | 3/3 | 3/3 |
| E05 | 2/3 | 3/3 | 3/3 | 3/3 |
| E06 | 3/3 | 3/3 | 3/3 | 3/3 |
| E07 | 3/3 | 3/3 | 3/3 | 3/3 |
| E08 | 3/3 | 3/3 | 3/3 | 3/3 |
| E09 | 3/3 | 3/3 | 3/3 | 3/3 |
| E10 | 3/3 | 3/3 | 3/3 | 3/3 |

## Cost

| metric | before | after |
|---|---|---|
| Cost per 100 tickets | $0.1052 | $0.1090 |
| Tokens in per ticket | 2622 | 2701 |
| Tokens out per ticket | 271 | 295 |
| Avg latency per ticket | 6788 ms | 6836 ms |
| Verifier verdicts | pass=11 | pass=12 |

Prices per 1M tokens: $0.36 in / $0.4 out.
