"""Language detection. Made-up sentences, not example or practice tickets."""

import pytest

from agent.lang import detect_language
from agent.models import Language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hoe kan ik de planning voor volgende week delen met mijn team?", Language.nl),
        ("Ik heb geen e-mail ontvangen, wat moet ik doen?", Language.nl),
        ("How do I share the schedule with my team for next week?", Language.en),
        ("The app does not open on my phone since yesterday.", Language.en),
        ("Wie kann ich den Dienstplan mit meinem Team teilen? Danke!", Language.de),
        ("Ich habe keine E-Mail bekommen, was muss ich tun?", Language.de),
    ],
)
def test_detects_short_samples(text: str, expected: Language) -> None:
    result = detect_language(text)
    assert result is not None
    assert result[0] is expected
    assert 0.7 <= result[1] <= 1.0


def test_english_with_dutch_product_words_is_english() -> None:
    text = (
        "Hi, where do I find Instellingen → Abonnement → Facturen? "
        "I can't see the Rooster → Verzoeken page either. Thanks"
    )
    result = detect_language(text)
    assert result is not None and result[0] is Language.en


@pytest.mark.parametrize("text", ["", "   ", "Hoi", "Help!", "Rooster export", "iOS 15 crash"])
def test_empty_or_very_short_is_none(text: str) -> None:
    assert detect_language(text) is None


def test_evenly_mixed_text_is_none() -> None:
    # Half Dutch, half English function words: too ambiguous to call.
    # nl: ik, heb, een, het (4) vs en: how, do, i, do, this (5) -> 56%, below the 70% bar.
    assert detect_language("Ik heb een vraag over het rooster. How do I do this?") is None
