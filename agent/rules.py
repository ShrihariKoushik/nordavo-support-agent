"""Deterministic safety layer. No model involved, so it cannot be talked out of a rule.

- `forced_handoff_hits`: client rules that always go to a human (nl/en/de vocabulary).
- `is_empty`: nothing but greetings, sign-offs and a name.
- `redact_payment_details`: mask card numbers and IBANs (used for logs too).
- `guess_language`: fallback nl/en/de when no model output is available.

The patterns are general vocabulary for each intent, not phrases from specific tickets.
They aim for precision; the LLM flags the same intents, so this is a safety net.
"""

import re
from enum import StrEnum

from pydantic import BaseModel


class Rule(StrEnum):
    refund_or_dispute = "refund_or_dispute"
    cancellation_or_contract = "cancellation_or_contract"
    gdpr_data_request = "gdpr_data_request"
    security_incident = "security_incident"
    legal_threat = "legal_threat"
    churn_threat = "churn_threat"


class RuleHit(BaseModel):
    rule: Rule
    matched_phrase: str


_POSSESSIVE = r"(?:my|our|the|this|your)"

_PATTERNS: dict[Rule, list[str]] = {
    Rule.refund_or_dispute: [
        # nl
        r"terugbetal\w*", r"geld\s+terug", r"terugstort\w*", r"restitutie", r"crediter\w*",
        r"creditnota", r"storner\w*",
        r"(?:dubbel|ten\s+onrechte|onterecht|te\s+veel)\s+"
        r"(?:afgeschreven|gefactureerd|betaald|in\s+rekening\s+gebracht|geïncasseerd)",
        # en
        r"refund\w*", r"money\s+back", r"reimburs\w*", r"charge-?back\w*", r"overcharg\w*",
        r"(?:charged|billed)\s+twice", r"double[\s-]+(?:charged|billed)",
        r"(?:wrongly|incorrectly)\s+(?:charged|billed)",
        rf"disput\w*\s+(?:{_POSSESSIVE}\s+)?(?:charge|payment|invoice|transaction)s?",
        # de
        r"r(?:ü|ue)ckerstatt\w*", r"erstattung", r"zur(?:ü|ue)ckerstatt\w*",
        r"geld\s+zur(?:ü|ue)ck", r"gutschrift", r"r(?:ü|ue)ckbuchung",
        r"doppelt\s+(?:abgebucht|berechnet|bezahlt|belastet)", r"zu\s+viel\s+berechnet",
    ],
    Rule.cancellation_or_contract: [
        # nl
        r"opzeg\w*", r"opgezegd",
        r"(?:abonnement|contract|account)\s+(?:stopzetten|be[eë]indigen|ontbinden|annuleren|"
        r"opheffen|wijzigen|aanpassen|overzetten|overdragen)",
        r"be[eë]indig\w*\s+(?:het|ons|mijn)\s+(?:abonnement|contract)",
        # en
        rf"cancel\w*\s+(?:{_POSSESSIVE}\s+)?(?:subscription|account|contract|plan|membership)",
        r"(?:subscription|contract|account)\s+cancel\w*",
        rf"terminat\w*\s+(?:{_POSSESSIVE}\s+)?(?:contract|subscription|agreement|account)",
        rf"(?:end|change)\s+{_POSSESSIVE}\s+(?:contract|subscription|agreement)",
        r"notice\s+period",
        r"downgrad\w*",
        # de
        r"k(?:ü|ue)ndig\w*", r"gek(?:ü|ue)ndigt",
        r"(?:vertrag|abo|abonnement)\s+(?:beenden|auflösen|aufl(?:ö|oe)sen|(?:ä|ae)ndern|wechseln)",
    ],
    Rule.gdpr_data_request: [
        # nl
        r"inzageverzoek\w*", r"verwijderverzoek\w*",
        r"inzage\s+in\s+(?:mijn|onze|alle|de)\s+(?:\w+\s+)?gegevens",
        r"(?:mijn|onze|alle|persoons)\s*gegevens\s+(?:laten\s+)?(?:verwijderen|wissen|inzien|opvragen)",
        r"recht\s+op\s+(?:vergetelheid|inzage|verwijdering|dataportabiliteit|gegevenswissing)",
        # en
        r"gdpr", r"dsar", r"data\s+subject\s+(?:access\s+)?request",
        r"(?:delete|erase|remove)\s+(?:all\s+)?(?:my|our|his|her|their)\s+(?:personal\s+)?data",
        r"right\s+to\s+(?:be\s+forgotten|erasure|access)",
        r"copy\s+of\s+(?:all\s+)?(?:my|our)\s+(?:personal\s+)?data",
        r"what\s+(?:personal\s+)?data\s+(?:do\s+)?you\s+(?:hold|have|store|keep)\s+(?:on|about)",
        # de
        r"dsgvo", r"auskunftsersuchen", r"datenauskunft",
        r"auskunft\s+(?:(?:ü|ue)ber|nach)\s+(?:meine|unsere|die)\s+(?:gespeicherten\s+)?daten",
        r"(?:meine|unsere|alle|personenbezogenen)\s+daten\s+(?:\w+\s+)?l(?:ö|oe)schen",
        r"recht\s+auf\s+(?:vergessenwerden|l(?:ö|oe)schung|auskunft)",
    ],
    Rule.security_incident: [
        # nl
        r"datalek\w*", r"gehackt", r"hacker\w*", r"phishing", r"onbevoegd\w*",
        r"verdachte\s+(?:inlog\w*|login\w*|activiteit\w*)",
        r"(?:account|wachtwoord|gegevens)\s+(?:is|zijn)\s+(?:overgenomen|gelekt|gestolen)",
        r"(?:gegevens|rooster|roosters|medewerkers|namen)\s+van\s+een\s+ander(?:e)?\s+"
        r"(?:bedrijf|klant|organisatie|restaurant|winkel|zaak)",
        # en
        r"data\s+(?:breach|leak)\w*", r"security\s+(?:breach|incident)\w*", r"hacked", r"hacker\w*",
        r"compromised", r"unauthori[sz]ed\s+(?:access|log-?\s?ins?|sign-?\s?ins?|user)",
        r"suspicious\s+(?:log-?\s?ins?|sign-?\s?ins?|activity|email)",
        r"(?:see|seeing|shows?|showing|visible)\s+(?:\w+\s+){0,4}"
        r"(?:another|other|a\s+different)\s+(?:company|customer|business|organi[sz]ation)",
        # de
        r"datenleck\w*", r"datenpanne\w*", r"hackerangriff\w*", r"kompromittiert",
        r"unbefugte[rnm]?\s+(?:zugriff|anmeldung|login|zugang)",
        r"verd(?:ä|ae)chtige[rn]?\s+(?:anmeldung|aktivit(?:ä|ae)t|login)\w*",
        r"(?:daten|dienstplan|dienstpläne|mitarbeiter)\s+(?:einer|eines)\s+anderen\s+"
        r"(?:firma|unternehmens|kunden|betriebs|restaurants)",
    ],
    Rule.legal_threat: [
        # nl
        r"advoca(?:at|ten)", r"jurist\w*", r"juridische\s+(?:stappen|actie|procedure)",
        r"rechtszaak\w*", r"rechtbank", r"(?:naar|bij|voor)\s+de\s+rechter",
        r"aansprakelijk\w*", r"ingebrekestelling", r"sommatie", r"schadevergoeding",
        r"autoriteit\s+persoonsgegevens", r"aangifte\s+doen",
        # en
        r"lawyers?", r"attorneys?", r"solicitors?", r"legal\s+(?:action|steps|proceedings)",
        r"lawsuit\w*", r"small\s+claims", r"cease\s+and\s+desist", r"breach\s+of\s+contract",
        r"(?:hold\s+you|be\s+held)\s+liable", r"(?:take|taking)\s+(?:you|this|it)\s+to\s+court",
        r"(?:will|going\s+to|gonna|'ll)\s+sue", r"sue\s+(?:you|roosterly|your\s+company)",
        # de
        r"anwalt\w*", r"anw(?:ä|ae)lt\w*", r"rechtsanw\w*", r"rechtliche\s+schritte",
        r"verklagen", r"klage\s+einreichen", r"vor\s+gericht", r"gerichtlich\w*",
        r"abmahnung", r"schadens?ersatz", r"haftbar",
    ],
    Rule.churn_threat: [
        # nl: "overstappen naar Plus" is an upgrade, not churn
        r"(?:overstappen|overstap|stappen\s+over|stap\s+over)\s+(?:naar|op)\s+"
        r"(?!(?:het\s+|de\s+|ons\s+|een\s+)?(?:plus|basis|abonnement|jaar|maand|nieuwe)\b)\w+",
        r"(?:naar|bij|voor)\s+een\s+(?:ander(?:e)?|concurrerende?)\s+"
        r"(?:aanbieder|leverancier|pakket|programma|software|app|systeem|partij|tool)",
        r"concurrent(?:en)?", r"(?:stoppen|vertrekken|weggaan)\s+(?:met|bij)\s+roosterly",
        # en
        r"switch\w*\s+to\s+(?:another|a\s+different|a\s+competing|a\s+competitor|your\s+competitor)",
        r"(?:move|moving|go|going)\s+to\s+(?:a\s+|your\s+)?competitors?",
        r"(?:leave|leaving|quit|quitting)\s+roosterly", r"other\s+(?:providers|vendors)",
        r"(?:look|looking|search|searching)\s+(?:for|at)\s+(?:an\s+|other\s+)?alternatives?\s+"
        r"(?:to|for)\s+(?:roosterly|you|your)",
        # de
        r"(?:anderen|neuen)\s+anbieter", r"anbieter\s*wechsel\w*", r"(?:zur|zu\s+der)\s+konkurrenz",
        r"roosterly\s+verlassen",
    ],
}

# "AVG" is the Dutch GDPR; case-sensitive so "avg hours" (average) does not match.
_CASE_SENSITIVE: dict[Rule, list[str]] = {Rule.gdpr_data_request: [r"AVG"]}


def _compile(patterns: list[str], flags: int) -> re.Pattern[str]:
    joined = "|".join(f"(?:{p})" for p in patterns)
    return re.compile(rf"(?<!\w)(?:{joined})(?!\w)", flags)


_COMPILED: list[tuple[Rule, re.Pattern[str]]] = [
    (rule, _compile(patterns, re.IGNORECASE)) for rule, patterns in _PATTERNS.items()
] + [(rule, _compile(patterns, 0)) for rule, patterns in _CASE_SENSITIVE.items()]


def forced_handoff_hits(subject: str, body: str) -> list[RuleHit]:
    """Every forced-handoff rule the ticket triggers, with the first phrase that matched."""
    text = " ".join(f"{subject}\n{body}".split())  # one space everywhere, so phrases span lines
    hits: dict[Rule, RuleHit] = {}
    for rule, pattern in _COMPILED:
        match = pattern.search(text)
        if match and rule not in hits:
            hits[rule] = RuleHit(rule=rule, matched_phrase=match.group())
    return [hits[rule] for rule in Rule if rule in hits]


# --- Empty tickets ---------------------------------------------------------------------------

_GREETINGS = [
    "hoi", "hallo", "hey", "hi", "hello", "dear", "beste", "geachte", "goedemorgen",
    "goedemiddag", "goedenavond", "goedendag", "good morning", "good afternoon", "good evening",
    "guten morgen", "guten tag", "guten abend", "moin", "servus", "sehr geehrte",
    "sehr geehrter", "liebe", "lieber", "damen und herren", "dames en heren", "heer", "mevrouw",
    "sir", "madam", "to whom it may concern", "team", "roosterly", "support", "helpdesk",
    "re", "fw", "fwd", "aw", "wg", "no subject", "geen onderwerp", "kein betreff",
]
_CLOSINGS = [
    "groet", "groeten", "gr", "mvg", "met vriendelijke groet", "met vriendelijke groeten",
    "vriendelijke groet", "vriendelijke groeten", "hartelijke groet", "hartelijke groeten",
    "regards", "kind regards", "best regards", "best", "cheers", "thanks", "thank you", "thx",
    "bedankt", "alvast bedankt", "dank", "dank je", "dankjewel", "dank u wel", "dank je wel",
    "mit freundlichen grüßen", "mfg", "viele grüße", "liebe grüße", "lg", "grüße", "gruß",
    "danke", "vielen dank", "sent from my iphone", "sent from my phone",
    "verzonden vanaf mijn iphone", "von meinem iphone gesendet",
]
_CLOSING_PHRASES = {tuple(phrase.split()) for phrase in _CLOSINGS}
# Longest first, so "met vriendelijke groet" wins over "groet".
_FILLER = sorted(
    {tuple(phrase.split()) for phrase in _GREETINGS} | _CLOSING_PHRASES, key=len, reverse=True
)
_WORD_RE = re.compile(r"\w+")


def is_empty(subject: str, body: str) -> bool:
    """True when there is nothing to answer: blank, or only greetings, sign-offs and a name.

    Conservative on purpose: a ticket we call empty may be closed, so any real word
    in the subject or body makes it non-empty.
    """
    if _leftover_words(subject)[0]:
        return False
    rest, saw_closing = _leftover_words(body)
    if not rest:
        return True
    # "Groeten, Jan de Vries": a short run of capitalised words after a sign-off is a name.
    return saw_closing and len(rest) <= 4 and all(_name_like(word) for word in rest)


_NAME_PARTICLES = {"de", "van", "der", "den", "ter", "te", "von", "zu", "du", "la", "le"}


def _name_like(word: str) -> bool:
    return word[:1].isupper() or word in _NAME_PARTICLES


def _leftover_words(text: str) -> tuple[list[str], bool]:
    words = _WORD_RE.findall(text)
    lower = [word.lower() for word in words]
    rest: list[str] = []
    saw_closing = False
    index = 0
    while index < len(words):
        phrase = _match_phrase(lower, index)
        if phrase:
            saw_closing = saw_closing or phrase in _CLOSING_PHRASES
            index += len(phrase)
        else:
            rest.append(words[index])
            index += 1
    return rest, saw_closing


def _match_phrase(lower: list[str], index: int) -> tuple[str, ...] | None:
    for phrase in _FILLER:
        if tuple(lower[index : index + len(phrase)]) == phrase:
            return phrase
    return None


# --- Language guess (only used when the model gives us nothing) -----------------------------

_LANGUAGE_WORDS: dict[str, set[str]] = {
    "nl": {"de", "het", "een", "ik", "we", "wij", "niet", "is", "en", "van", "voor", "hoe", "mijn",
           "ons", "onze", "kan", "kunnen", "graag", "bedankt", "groet", "dat", "dit", "met", "op"},
    "en": {"the", "a", "an", "i", "we", "not", "is", "and", "of", "for", "how", "my", "our", "can",
           "please", "thanks", "regards", "that", "this", "with", "on", "to"},
    "de": {"der", "die", "das", "ein", "eine", "ich", "wir", "nicht", "ist", "und", "von", "für",
           "wie", "mein", "unser", "kann", "bitte", "danke", "grüße", "dass", "mit", "auf"},
}


def guess_language(subject: str, body: str) -> str:
    """Rough nl/en/de guess by common words. Dutch on a tie: most customers write Dutch."""
    words = [word.lower() for word in _WORD_RE.findall(f"{subject} {body}")]
    scores = {lang: sum(word in vocab for word in words) for lang, vocab in _LANGUAGE_WORDS.items()}
    best = max(scores, key=lambda lang: scores[lang])
    return best if scores[best] > scores["nl"] else "nl"


# --- Payment details --------------------------------------------------------------------------

CARD_MASK = "[REDACTED_CARD]"
IBAN_MASK = "[REDACTED_IBAN]"

# 13-19 digits, single spaces or dashes allowed between digits.
_CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
# Country code, 2 check digits, then letters/digits, optionally in groups of 4.
# No checksum check: a mistyped IBAN is still masked.
_IBAN_RE = re.compile(
    r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b",
    re.IGNORECASE,
)


def redact_payment_details(text: str) -> str:
    """Mask IBANs and Luhn-valid card numbers. Other numbers stay as they are."""
    text = _IBAN_RE.sub(_mask_iban, text)
    return _CARD_RE.sub(_mask_card, text)


def _mask_iban(match: re.Match[str]) -> str:
    compact = match.group().replace(" ", "")
    # Real IBANs are 15-34 characters. Shorter matches are something else.
    return IBAN_MASK if 15 <= len(compact) <= 34 else match.group()


def _mask_card(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group())
    return CARD_MASK if _luhn_ok(digits) else match.group()


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0
