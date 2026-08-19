"""Convert an ASL gloss sequence into a natural English sentence.

Primary path: OpenRouter (OpenAI-compatible /chat/completions) using the model in
config.OPENROUTER_MODEL. Falls back to a small rule-based rewriter whenever there
is no API key, no network, or the request errors — so a demo never hard-fails.

The LLM here is a *downstream text rewriter*, not the recognizer (see spec §7).
"""
from __future__ import annotations

import httpx

import config

_SYSTEM_PROMPT = (
    "You convert American Sign Language (ASL) gloss sequences into natural, "
    "grammatically correct English sentences. Reply with ONLY the sentence, no "
    "quotes, no explanation. Keep it short and faithful to the gloss meaning."
)


def _clean_gloss(gloss: list[str]) -> list[str]:
    return [config.display_word(w).lower() for w in gloss if w]


def rule_based(gloss: list[str]) -> str:
    """Lightweight offline rewriter: enough to be intelligible for a demo.

    Handles a few common ASL patterns (ME -> I, dropped articles/copula) and
    falls back to capitalized, space-joined gloss with a period.
    """
    words = _clean_gloss(gloss)
    if not words:
        return ""

    replace = {"me": "I", "you": "you", "thank you": "thank you"}
    words = [replace.get(w, w) for w in words]

    # naive article insertion before common nouns
    nouns = {"water", "food", "home", "help"}
    out: list[str] = []
    for i, w in enumerate(words):
        if w in nouns and (i == 0 or words[i - 1] not in {"the", "some", "a"}):
            out.append("some" if w in {"water", "food"} else "the")
        out.append(w)

    sentence = " ".join(out).strip()
    sentence = sentence[0].upper() + sentence[1:]
    if sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _openrouter(gloss: list[str]) -> str:
    gloss_str = " ".join(config.display_word(w) for w in gloss)
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Sign Language Translator",
    }
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Convert this ASL gloss into a natural English sentence. "
                        f"Gloss: {gloss_str}"},
        ],
        "temperature": 0.3,
        "max_tokens": 60,
    }
    url = config.OPENROUTER_BASE_URL.rstrip("/") + "/chat/completions"
    resp = httpx.post(url, headers=headers, json=payload,
                      timeout=config.OPENROUTER_TIMEOUT_S)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    return text.strip('"').strip()


def gloss_to_sentence(gloss: list[str]) -> tuple[str, str]:
    """Return (sentence, source) where source is 'openrouter' or 'rule-based'."""
    if not gloss:
        return "", "rule-based"
    if config.OPENROUTER_API_KEY:
        try:
            return _openrouter(gloss), "openrouter"
        except Exception as e:  # network/key/model errors -> graceful fallback
            print(f"[gloss_to_sentence] OpenRouter failed ({e}); using rule-based")
    return rule_based(gloss), "rule-based"


if __name__ == "__main__":
    import sys
    demo = [w.upper() for w in (sys.argv[1:] or ["ME", "WANT", "WATER"])]
    sentence, source = gloss_to_sentence(demo)
    print(f"gloss:    {' '.join(demo)}")
    print(f"sentence: {sentence}   [{source}]")
