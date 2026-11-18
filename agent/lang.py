"""Detect nl/en/de from common function words. No dependency, fully explainable.

Product words are Dutch in every language ("Instellingen", "Rooster"), so we only count
short function words ("the", "het", "und") that a customer writes in their own language.
Words used in more than one of the languages ("is", "was", "die", "we") are not counted.
"""

import re

from agent.models import Language

_WORDS: dict[Language, set[str]] = {
    Language.nl: set(
        """de het een en ik je jij u wij zij niet geen van voor met op aan bij naar uit om
        dat dit deze wat hoe waar wanneer waarom maar ook nog al wel kan kunnen moet wil
        willen heb hebben heeft zijn wordt worden mijn ons onze jullie er daar graag bedankt
        alvast vriendelijke groet groeten goedemorgen hoi dank als te dan mee nu toch even
        niets iets is we die hier wie of zo hallo""".split()
    ),
    Language.en: set(
        """the a an and i you we our my is are was were be been not no of for with on at to
        from this that these those what how where when why but also can could would should
        will have has had do does did don it its there here please thanks thank regards hi
        hello if or so then just other any some get got me your""".split()
    ),
    Language.de: set(
        """der die das ein eine einen einem und ich sie wir unser unsere mein meine nicht
        kein keine von für mit auf an bei nach aus zu dass dies diese was wie wo wann warum
        aber auch noch schon kann können muss will möchte habe haben hat ist sind wird
        werden es hier da bitte danke grüße hallo guten tag wenn oder so dann nur mir ihr
        ihnen uns im dem den des als""".split()
    ),
}

# A word in two vocabularies says nothing about which language it is.
_SHARED = {
    word
    for lang, words in _WORDS.items()
    for word in words
    if any(word in other for other_lang, other in _WORDS.items() if other_lang is not lang)
}
_SIGNAL = {lang: words - _SHARED for lang, words in _WORDS.items()}

MIN_HITS = 3  # fewer marker words than this is too short to call
MIN_SHARE = 0.7  # the winning language needs at least this share of all marker words

_TOKEN_RE = re.compile(r"[^\W\d_]+")


def detect_language(text: str) -> tuple[Language, float] | None:
    """Return (language, confidence) or None when the text is too short or mixed.

    Confidence is the winner's share of all counted marker words (0.7-1.0 when returned).
    """
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    counts = {lang: sum(token in words for token in tokens) for lang, words in _SIGNAL.items()}
    total = sum(counts.values())
    if total < MIN_HITS:
        return None
    best = max(counts, key=lambda lang: counts[lang])
    share = counts[best] / total
    if share < MIN_SHARE:
        return None
    return best, round(share, 2)
