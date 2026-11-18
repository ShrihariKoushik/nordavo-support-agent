# Roosterly support agent

Reads every open ticket from the Roosterly helpdesk and posts exactly one decision per ticket:
`reply` (in the customer's language), `handoff` (English note for a human) or `close`.

## How to run

```
git clone https://github.com/ShrihariKoushik/nordavo-support-agent.git && cd nordavo-support-agent
python -m venv .venv && .venv\Scripts\activate        # Windows; source .venv/bin/activate elsewhere
pip install -r requirements.txt
cp .env.example .env                                  # fill in HELPDESK_KEY, CANDIDATE_EMAIL, LLM_API_KEY

python -m agent status                                # mode, open / handled counts
python -m agent run --dry-run                         # decide everything, post nothing, write no journal
python -m agent run --dry-run --ticket T004           # only these tickets (repeatable)
python -m agent run                                   # the real run
python -m agent reset                                 # re-open practice tickets, archive the journal
python -m eval.run_eval --repeats 3 --tag NAME        # score decide() on the 10 labelled examples
python -m eval.compare BEFORE AFTER                   # writes eval/results/compare-BEFORE-AFTER.md
python -m pytest -q
```

## Decisions

1. **The model proposes, code decides.** Deterministic rules (`agent/rules.py`, nl/en/de) force a
   handoff for refunds, cancellations/contract changes, GDPR requests, security incidents, legal and
   churn threats, and instructions aimed at the agent (prompt injection). A reply also needs a real FAQ
   id and confidence ≥ 0.75. Code blocks promises to the customer in a reply ("your request is on the
   roadmap") and claimed actions in handoff notes ("added to", "discount").
2. **An independent check of every reply.** A separate LLM call sees only the ticket, the draft and the
   cited FAQ text. It back-translates the draft to English and checks each claim against the FAQ. Code
   decides pass/fail from that, and code's language detection overrules the checker on language.
   I don't read Dutch, so I work in Dutch directly and audit through the back-translation, which is
   logged with every reply. The ticket language is detected in code (`agent/lang.py`), not taken from the model.
3. **Qwen 2.5 72B via OpenRouter**, pinned to one provider (`deepinfra/fp8`, no fallbacks) with a fixed
   seed and temperature 0. About $0.12 per 100 tickets (calculation under Eval and fix).

**AI tools.** I started in Cursor and switched to Claude Code when I hit the Cursor limit. Both read
`AGENTS.md` and `.cursor/rules`. A second AI assistant reviewed each commit independently (tests, live
read-only checks).

**Assumptions.** Handoff notes and `reason` are in English. A forced handoff is at least P2 (money,
contract, GDPR); an injection ticket is exactly P2, whatever the ticket claims. Only empty tickets
(decided in code) or spam with confidence ≥ 0.75 are closed. Labels have no FAQ field, so the eval
reads the expected FAQ id from the label's reason text. Explaining how to upgrade a plan
(F05, e.g. T023) is treated as a how-to, not a contract change: the agent changes nothing for the
customer, it only points to the self-service setting. Cancellations, downgrades and other changes to
the contract always go to a human.

## Reliability

Requests are paced to 4 per second. A 503 is retried with full-jitter backoff (5 attempts), as is a
GET timeout; a 429 waits for `Retry-After`. A 409 is never retried. If a POST loses its response, the
agent GETs the ticket and posts again only if it is still open. A true 409 and "probably ours" are
logged separately. An append-only journal (`state/journal.jsonl`) records `decided` before each POST
and `posted` after it, so a re-run skips handled tickets and re-checks unfinished ones with a GET. A
half-written last line is ignored. Code redacts card numbers (Luhn-checked) and IBANs before anything
is posted or logged. Keys come only from the environment. Ticket text is wrapped as untrusted data and is
never followed as instructions.

## Eval and fix

All results are in [`eval/results/`](eval/results/), for example [baseline](eval/results/baseline.md)
and [final4](eval/results/final4.md).
**The one fix:** English tickets were labelled `nl` (E01 0/3, E06 1/3), so language is now detected
in code. Language accuracy went from 87% to 100%, and unsafe replies stayed at 0
([compare-baseline-after-fix](eval/results/compare-baseline-after-fix.md)).

The 30-ticket dry run then showed problems the eval set did not have. Each was fixed with a test:
a prompt injection (T030), Dutch back-translations, handoff notes claiming actions, invented customer
names, and promises in replies. Unsafe replies were 0 in every eval run.

**Variation.** The same code gives different results between runs. `final4` and `final4b` use
identical code and gave action 77% vs 80% and FAQ 61% vs 67%
([compare-final4-final4b](eval/results/compare-final4-final4b.md)). E02 alone has scored 0/3 to 2/3
across runs with an identical prompt. Compare runs with that in mind.

**Final run:** 30 tickets posted (8 reply, 20 handoff, 2 close), 0 errors. The log is in
[`logs/`](logs/). A second dry run found nothing open, and the journal skips all 30.

**Cost per 100 tickets** (final run, Qwen 2.5 72B at $0.36 in / $0.40 out per 1M tokens):
87,461 in × $0.36/1M = $0.0315; 9,294 out × $0.40/1M = $0.0037; total $0.0352 for 30 tickets,
so $0.0352 / 30 × 100 ≈ **$0.12 per 100 tickets** (reply + verifier calls included). The whole
assignment (every eval, dry run and the final run) cost $0.35 on OpenRouter, well under €1.

## What I would improve

**Next step:** cut needless handoffs without loosening safety, measured with more repeats.

- Fewer needless handoffs: E01 (old iOS, F12), the feature requests T004/T016/T020 (F10), and T015,
  where the checker is too strict.
- Contract over-flagging by the model: E02 (VAT number), T008 (new card).
- Priority drift on how-to questions (T017 got P2).
- Grammar repair for replies ("elke idee" → "elk idee"); the checker does not report it.
- More labelled examples and more repeats, so a fix can be told apart from run-to-run variation.
- An injection note that keeps the underlying reason: T007 was a refund and an injection, and the
  human only gets the injection note.

**One thing I would change in the client's rules:** drop "most customers write in Dutch" and keep only
"reply in the customer's language". That sentence pushed the model to label English tickets as Dutch
(E01/E06, the fix above). In the brief, I would have `GET /tickets/{id}` return the stored decision, so
a posted reply or note can be audited after the run.
