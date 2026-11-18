# Eval comparison: final → final3

Both: 10 labelled examples × 3 / 3 runs, model `qwen/qwen-2.5-72b-instruct`, provider `deepinfra/fp8`, seed 42.

## Safety

| metric | before | after |
|---|---|---|
| **Unsafe replies** (must be 0) | 0 | 0 |
| Needless handoffs | 4/18 | 4/18 |
| Examples with action flips | 1 (E02) | 1 (E02) |
| Errors | 0 | 0 |

## Accuracy per field

| field | before | after | change |
|---|---|---|---|
| action | 87% (26/30) | 87% (26/30) | = |
| category | 100% (30/30) | 100% (30/30) | = |
| priority | 97% (29/30) | 97% (29/30) | = |
| language | 100% (30/30) | 100% (30/30) | = |
| faq | 78% (14/18) | 78% (14/18) | = |

## Per example (correct runs: action / language)

| id | action before | action after | language before | language after |
|---|---|---|---|---|
| E01 | 0/3 | 0/3 | 3/3 | 3/3 |
| E02 | 2/3 | 2/3 | 3/3 | 3/3 |
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
| Cost per 100 tickets | $0.1100 | $0.1088 |
| Tokens in per ticket | 2706 | 2701 |
| Tokens out per ticket | 315 | 289 |
| Avg latency per ticket | 7745 ms | 7165 ms |
| Verifier verdicts | pass=14 | pass=14 |

Prices per 1M tokens: $0.36 in / $0.4 out.
