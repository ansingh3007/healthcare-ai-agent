"""
input_guard.py — Filter unsafe or out-of-scope inputs.
"""

BLOCKED_PATTERNS = [
    "specific dosage for me",
    "should i take",
    "diagnose me",
    "do i have",
    "am i sick",
    "personal medical advice",
    "my symptoms",
    "ignore previous instructions",
    "system prompt",
    "jailbreak",
    "act as a doctor",
]

OUT_OF_SCOPE_TOPICS = [
    "legal advice",
    "financial advice",
    "personal relationships",
    "non-medical",
]

DISCLAIMER = (
    "\n\n⚠️ This system provides population-level healthcare data and "
    "clinical guidelines only. It does not provide personal medical advice. "
    "Please consult a qualified healthcare professional for personal health decisions."
)


def check_input(text: str) -> dict:
    """Check if input is safe and in scope. Returns {safe, reason, modified_text}."""
    text_lower = text.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in text_lower:
            return {
                "safe": False,
                "reason": f"Input requests personal medical advice ('{pattern}'). "
                          "This system provides population-level data and guidelines only.",
                "modified_text": None,
            }

    for topic in OUT_OF_SCOPE_TOPICS:
        if topic in text_lower:
            return {
                "safe": False,
                "reason": f"Topic '{topic}' is outside the scope of this healthcare data system.",
                "modified_text": None,
            }

    return {"safe": True, "reason": None, "modified_text": text}
