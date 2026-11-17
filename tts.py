"""Text-to-speech synthesis helpers."""

from io import BytesIO
import re
from typing import Optional, Tuple

from gtts import gTTS


def _clean_text_for_tts(text: str) -> str:
    """Remove common Markdown syntax while preserving its readable text."""
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    cleaned = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*>[ \t]?", "", cleaned)
    cleaned = re.sub(r"(?m)^[ \t]*[-*][ \t]+", "", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"(\*\*|__)(.+?)\1", r"\2", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", cleaned)
    cleaned = cleaned.replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def synthesize_speech(
    text: str, lang: str = "en"
) -> Tuple[Optional[bytes], Optional[str]]:
    """Synthesize speech to MP3 bytes, returning any error instead of raising."""
    clean_text = _clean_text_for_tts(text or "")
    if not clean_text:
        return None, "Text for speech synthesis is empty."

    try:
        audio = BytesIO()
        gTTS(text=clean_text, lang=lang).write_to_fp(audio)
        return audio.getvalue(), None
    except Exception as e:
        return None, f"Speech synthesis failed: {e}"
