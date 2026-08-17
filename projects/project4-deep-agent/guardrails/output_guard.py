"""
output_guard.py — Ensure all outputs include safety disclaimers.
"""

REQUIRED_DISCLAIMER = (
    "\n\n---\n*Clinical decisions should always involve qualified healthcare professionals. "
    "This report is generated from healthcare data and guidelines for informational purposes only.*"
)

DANGEROUS_PHRASES = [
    "you should take",
    "i recommend taking",
    "the correct dose for you",
    "you have",
    "you are suffering from",
    "self-medicate",
]


def check_output(text: str) -> dict:
    """Check output safety and add disclaimer if needed."""
    text_lower = text.lower()
    warnings = []

    for phrase in DANGEROUS_PHRASES:
        if phrase in text_lower:
            warnings.append(f"Output may contain personal medical advice: '{phrase}'")

    # Always append disclaimer if not already present
    if "healthcare professional" not in text_lower:
        safe_text = text + REQUIRED_DISCLAIMER
    else:
        safe_text = text

    return {
        "safe": len(warnings) == 0,
        "warnings": warnings,
        "text": safe_text,
    }
