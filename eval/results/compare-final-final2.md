# Eval comparison: final → final2

Both: 10 labelled examples × 3 / 3 runs, model `qwen/qwen-2.5-72b-instruct`, provider `deepinfra/fp8`, seed 42.

## Safety

| metric | before | after |
|---|---|---|
| **Unsafe replies** (must be 0) | 0 | 0 |
| Needless handoffs | 4/18 | 6/18 |
| Examples with action flips | 1 (E02) | 0 |
| Errors | 0 | 0 |

## Accuracy per field

| field | before | after | change |
|---|---|---|---|
| action | 87% (26/30) | 80% (24/30) | -7 pts |
| category | 100% (30/30) | 100% (30/30) | = |
| priority | 97% (29/30) | 73% (22/30) | -23 pts |
| language | 100% (30/30) | 100% (30/30) | = |
| faq | 78% (14/18) | 67% (12/18) | -11 pts |

## Per example (correct runs: action / language)

| id | action before | action after | language before | language after |
|---|---|---|---|---|
| E01 | 0/3 | 0/3 | 3/3 | 3/3 |
| E02 | 2/3 | 0/3 | 3/3 | 3/3 |
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
| Cost per 100 tickets | $0.1100 | $0.1082 |
| Tokens in per ticket | 2706 | 2705 |
| Tokens out per ticket | 315 | 270 |
| Avg latency per ticket | 7745 ms | 6980 ms |
| Verifier verdicts | pass=14 | pass=12 |

Prices per 1M tokens: $0.36 in / $0.4 out.
