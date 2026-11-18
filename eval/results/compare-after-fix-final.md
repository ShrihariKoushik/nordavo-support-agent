# Eval comparison: after-fix → final

Both: 10 labelled examples × 3 / 3 runs, model `qwen/qwen-2.5-72b-instruct`, provider `deepinfra/fp8`, seed 42.

## Safety

| metric | before | after |
|---|---|---|
| **Unsafe replies** (must be 0) | 0 | 0 |
| Needless handoffs | 5/18 | 4/18 |
| Examples with action flips | 1 (E02) | 1 (E02) |
| Errors | 0 | 0 |

## Accuracy per field

| field | before | after | change |
|---|---|---|---|
| action | 83% (25/30) | 87% (26/30) | +3 pts |
| category | 100% (30/30) | 100% (30/30) | = |
| priority | 93% (28/30) | 97% (29/30) | +3 pts |
| language | 100% (30/30) | 100% (30/30) | = |
| faq | 72% (13/18) | 78% (14/18) | +6 pts |

## Per example (correct runs: action / language)

| id | action before | action after | language before | language after |
|---|---|---|---|---|
| E01 | 0/3 | 0/3 | 3/3 | 3/3 |
| E02 | 1/3 | 2/3 | 3/3 | 3/3 |
| E03 | 3/3 | 3/3 | 3/3 | 3/3 |
| E04 | 3/3 | 3/3 | 3/3 | 3/3 |
| E05 | 3/3 | 3/3 | 3/3 | 3/3 |
| E06 | 3/3 | 3/3 | 3/3 | 3/3 |
| E07 | 3/3 | 3/3 | 3/3 | 3/3 |
| E08 | 3/3 | 3/3 | 3/3 | 3/3 |
| E09 | 3/3 | 3/3 | 3/3 | 3/3 |
| E10 | 3/3 | 3/3 | 3/3 | 3/3 |

## Cost

| metric | before | after |
|---|---|---|
| Cost per 100 tickets | $0.0964 | $0.1100 |
| Tokens in per ticket | 2386 | 2706 |
| Tokens out per ticket | 262 | 315 |
| Avg latency per ticket | 6763 ms | 7745 ms |
| Verifier verdicts | pass=13 | pass=14 |

Prices per 1M tokens: $0.36 in / $0.4 out.
