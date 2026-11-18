"""Deterministic safety layer. No model involved, so it cannot be talked out of a rule.

- `forced_handoff_hits`: client rules that always go to a human (nl/en/de vocabulary).
- `is_empty`: nothing but greetings, sign-offs and a name.
- `redact_payment_details`: mask card numbers and IBANs (used for logs too).

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
    # Not a client rule: text that tries to instruct the agent (prompt injection).
    instructions_in_ticket = "instructions_in_ticket"


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
    # Text aimed at the agent rather than at support. "Ignore my previous email" is normal
    # customer language, so "ignore" only counts when the object is instructions or rules.
    Rule.instructions_in_ticket: [
        # en
        r"special\s+instructions?",
        r"(?:ignore|disregard|forget)\s+(?:all\s+|any\s+|your\s+|the\s+)?(?:previous\s+|prior\s+|above\s+)?"
        r"(?:instructions?|rules|prompts?|guidelines)",
        r"system\s*prompt",
        r"(?:note|message|instructions?)\s+(?:for|to)\s+(?:the\s+)?(?:ai|bot|assistant|agent|model)\b",
        r"as\s+(?:the|an|a)\s+(?:ai|agent|assistant|bot|language\s+model)\b",
        r"you\s+must\s+(?:confirm|approve|grant|give|apply|promise|agree)",
        r"set\s+(?:the\s+|this\s+)?priority",
        # nl
        r"speciale\s+instructies?",
        r"negeer\s+(?:alle\s+|je\s+|jouw\s+|de\s+)?(?:eerdere\s+|vorige\s+|bovenstaande\s+)?"
        r"(?:instructies|regels|opdrachten|richtlijnen|prompts?)",
        r"systeem\s*prompt",
        r"(?:notitie|bericht|instructies?|opdracht)\s+(?:voor|aan)\s+(?:de\s+)?(?:ai|bot|assistent|agent)\b",
        r"(?:voor|aan)\s+de\s+(?:ai|ki)[\s-]*(?:assistent|agent|bot)",
        r"(?:zet|stel)\s+(?:de\s+)?prioriteit",
        # "bevestig ... korting" is a demand; "geeft Roosterly korting?" is a price question.
        r"bevestig\w*\s+(?:\w+\s+){0,4}korting",
        # de
        r"spezielle\s+anweisungen", r"systemprompt",
        r"ignorier\w*\s+(?:alle\s+|deine\s+|die\s+)?(?:vorherigen\s+|bisherigen\s+)?"
        r"(?:anweisungen|regeln|instruktionen|vorgaben)",
        r"(?:notiz|nachricht|anweisung\w*)\s+(?:f(?:ü|ue)r|an)\s+(?:die\s+|den\s+)?(?:ki|ai|assistent\w*|bot|agent\w*)\b",
        r"setze?\s+(?:die\s+)?priorit(?:ä|ae)t",
        r"best(?:ä|ae)tig\w*\s+(?:\w+\s+){0,4}rabatt",
        # any language: a percentage discount demand, or VIP status next to an instruction
        r"\d+\s?%\s*(?:korting|discount|rabatt)",
        r"vip\b.{0,200}?(?:instruct|anweisung|confirm|bevestig|best(?:ä|ae)tig|"
        r"prioriteit|priority|priorit(?:ä|ae)t|korting|discount|rabatt)\w*",
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


# --- Commitments in handoff notes ------------------------------------------------------------

# A handoff note tells a human what to check. It must never claim we did, promised or granted
# something: nothing has been sent and nothing has been decided.
_COMMITMENT_RE = re.compile(
    r"(?<!\w)(?:"
    r"added\s+to|we\s+(?:have\s+)?(?:fixed|added|resolved|refunded|credited|promised)|"
    r"(?:we|i|support)\s+(?:have\s+|has\s+)?provided|"
    r"provided\s+(?:a|an|the)\s+(?:workaround|solution|fix|instructions?|link|answer)|"
    r"(?:we|i|is|has\s+been|have\s+been)\s+confirmed|confirm(?:ed)?\s+(?:the\s+|a\s+)?\d*\s?%?\s*discount|"
    r"granted|discount|refund\s+(?:has\s+been\s+)?issued|issued\s+a\s+refund|promised|"
    r"toegevoegd|korting|toegezegd|terugbetaald|"
    r"hinzugef(?:ü|ue)gt|rabatt|gew(?:ä|ae)hrt|erstattet"
    r")(?!\w)",
    re.IGNORECASE,
)


def commitment_phrases(text: str) -> list[str]:
    """Phrases in a handoff note that claim an action or promise. Empty list if none."""
    return [match.group() for match in _COMMITMENT_RE.finditer(text)]


# --- Promises in a reply to the customer ------------------------------------------------------

# Only wording about THIS customer's request ("your request is on the roadmap", "we have
# forwarded it"). The FAQ's general wording ("feature requests go on our roadmap") must pass.
_REQUEST = r"(?:verzoek|idee|suggestie|aanvraag|wens|feedback)"
_REQUEST_EN = r"(?:request|idea|suggestion|feedback|wish)"
_REQUEST_DE = r"(?:anfrage|idee|vorschlag|wunsch|feedback)"
_PROMISE_RE = re.compile(
    r"(?<!\w)(?:"
    # nl
    rf"(?:je|jouw|uw)\s+{_REQUEST}\s+(?:staat|is|wordt|werd|komt)|"
    rf"(?:je|jouw|uw)\s+{_REQUEST}\s+(?:\w+\s+){{0,3}}(?:toegevoegd|genoteerd|doorgestuurd|meegenomen)|"
    r"(?:is|wordt|werd)\s+toegevoegd\s+aan\s+(?:onze|de)\s+(?:\w+\s+)?roadmap|"
    r"hebben\s+(?:we|wij)\s+(?:\w+\s+){0,3}(?:genoteerd|doorgestuurd|doorgegeven|toegevoegd)|"
    r"(?:we|wij)\s+hebben\s+(?:\w+\s+){0,3}(?:genoteerd|doorgestuurd|doorgegeven|toegevoegd)|"
    r"doorgestuurd|doorgezet|"
    # "we voegen deze toe" / "we zetten het op": a pronoun means THIS customer's request
    r"(?:we|wij|en)\s+(?:voegen|zetten|plaatsen|nemen)\s+(?:deze|dit|het|hem|die)\s+(?:\w+\s+){0,4}"
    r"(?:toe|op|mee)|"
    r"(?:we|wij)\s+werken\s+(?:er\s+)?(?:aan|eraan)|"
    r"(?:we|wij)\s+(?:gaan|zullen)\s+(?:\w+\s+){0,3}(?:bekijken|oppakken|oplossen|onderzoeken|toevoegen)|"
    # en
    rf"your\s+{_REQUEST_EN}\s+(?:is|has\s+been|was|will\s+be)|"
    r"(?:we|i|and)\s*(?:'ll\s+|will\s+)?(?:add|put|place|log)\s+(?:it|this|them|your\s+\w+)\s+(?:to|on|in)|"
    rf"(?:it|this)\s+(?:has\s+been|is\s+now)\s+(?:added|forwarded|noted|logged|passed\s+on|registered)|"
    r"(?:we|i)\s*(?:have|'ve)\s+(?:\w+\s+){0,2}(?:noted|forwarded|passed\s+on|added|logged|registered|escalated)|"
    r"(?:we|i)\s*(?:are|'re)\s+(?:already\s+|currently\s+|now\s+)?(?:working\s+on|looking\s+into)|"
    r"(?:we|i)\s*(?:will|'ll)\s+(?:\w+\s+){0,2}(?:look\s+into|fix|add|forward|pass\s+(?:it|this)\s+on|get\s+back)|"
    # de
    rf"(?:ihr|ihre|dein|deine)\s+{_REQUEST_DE}\s+(?:\w+\s+){{0,3}}(?:ist|wurde|wird|steht)|"
    r"wir\s+haben\s+(?:\w+\s+){0,3}(?:notiert|weitergeleitet|aufgenommen|hinzugef(?:ü|ue)gt|vermerkt)|"
    r"weitergeleitet|"
    r"(?:wir|und)\s+(?:f(?:ü|ue)gen|nehmen|setzen)\s+(?:sie|es|ihn|diese|diesen|dies|ihre\s+\w+|deine\s+\w+)\s+"
    r"(?:\w+\s+){0,4}(?:hinzu|auf)|"
    r"wir\s+arbeiten\s+(?:bereits\s+|schon\s+)?(?:daran|an)|"
    r"wir\s+werden\s+(?:\w+\s+){0,3}(?:pr(?:ü|ue)fen|beheben|hinzuf(?:ü|ue)gen|weiterleiten|umsetzen)"
    r")(?!\w)",
    re.IGNORECASE,
)


def promise_phrases(text: str) -> list[str]:
    """Phrases in a reply that tell the customer something was done or decided for them."""
    return [match.group() for match in _PROMISE_RE.finditer(" ".join(text.split()))]


# --- Names in a reply's greeting -------------------------------------------------------------

NEUTRAL_GREETING = {"nl": "Hallo,", "en": "Hello,", "de": "Hallo,"}

_NAME = r"[A-ZÀ-Þ][\w'-]*(?:\s+(?:van|de|der|den|von|ter|te|du|la)){0,2}(?:\s+[A-ZÀ-Þ][\w'-]*)?"
# "Hoi Jan," / "Goedemorgen Anouk!" / "Dear Mr Smith," / "Sehr geehrte Frau Becker,"
_GREETING_NAME_RE = re.compile(
    r"^(?P<greet>\s*(?i:hoi|hallo|hey|hi|hello|dear|beste|geachte|goedemorgen|goedemiddag|"
    r"goedenavond|guten\s+(?:tag|morgen|abend)|liebe[rs]?|sehr\s+geehrte[rs]?)"
    r"(?:\s+(?i:heer|mevrouw|meneer|mr\.?|mrs\.?|ms\.?|frau|herr))?)"
    rf"\s+(?P<name>{_NAME})(?P<end>\s*[,!.:]?)"
)
# "Bedankt voor je suggestie, Jan!" in the first line, when there is no greeting.
_VOCATIVE_NAME_RE = re.compile(r",\s*(?P<name>[A-ZÀ-Þ][\w'-]+)(?=\s*[!.,])")
# Words a greeting can address that are not a person's name.
_NOT_A_NAME = {"team", "roosterly", "sie", "allemaal", "iedereen", "there", "all", "everyone", "support"}
_PARTICLES = {"van", "de", "der", "den", "von", "ter", "te", "du", "la"}


def remove_invented_name(draft: str, ticket_text: str, language: str) -> tuple[str, str | None]:
    """Drop a name from the reply's opening if that name is not in the ticket.

    Returns the (possibly changed) draft and the removed name, or None if nothing changed.
    Only the first line is checked; a name the customer wrote themselves is kept.
    """
    ticket_words = {word.lower() for word in _WORD_RE.findall(ticket_text)}
    first_line, newline, rest = draft.partition("\n")

    match = _GREETING_NAME_RE.match(first_line)
    if match:
        name = match.group("name")
        if _name_in_ticket(name, ticket_words):
            return draft, None
        neutral = NEUTRAL_GREETING.get(language, "Hello,")
        tail = first_line[match.end():].lstrip(" ,!.:")
        new_first = f"{neutral} {tail}" if tail else neutral
        return new_first + newline + rest, name

    vocative = _VOCATIVE_NAME_RE.search(first_line)
    if vocative and not _name_in_ticket(vocative.group("name"), ticket_words):
        new_first = first_line[: vocative.start()] + first_line[vocative.end("name"):]
        return new_first + newline + rest, vocative.group("name")
    return draft, None


def _name_in_ticket(name: str, ticket_words: set[str]) -> bool:
    tokens = [token.lower() for token in name.split() if token.lower() not in _PARTICLES]
    if not tokens or all(token in _NOT_A_NAME for token in tokens):
        return True  # "Hallo team," is not a name; leave it alone
    return all(token.strip("'-.") in ticket_words for token in tokens)


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
