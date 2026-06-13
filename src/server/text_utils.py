"""
Text utilities for voice-friendly output.

Cleans AI responses for TTS (removes markdown, hashtags, etc.)
while preserving original for display.
"""

import re

_RE_CODE_BLOCK = re.compile(r"```[\s\S]*?```")
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_HEADER = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
_RE_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_RE_ITALIC = re.compile(r"\*([^*]+)\*")
_RE_UNDERLINE_BOLD = re.compile(r"__([^_]+)__")
_RE_UNDERLINE_ITALIC = re.compile(r"_([^_]+)_")
_RE_HASHTAG = re.compile(r"#(\w+)")
_RE_URL = re.compile(r"https?://\S+")
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_EMOJI = re.compile(r"[🔗📦📁💻🖥️⚡🔧🛠️📝✅❌⚠️🚀🎯💡🔍📊📈📉🗂️📋]")
_RE_BULLET = re.compile(r"^\s*[-•]\s*", flags=re.MULTILINE)
_RE_NUMBERED = re.compile(r"^\s*\d+\.\s*", flags=re.MULTILINE)
_RE_MULTI_NEWLINE = re.compile(r"\n{2,}")
_RE_NEWLINE = re.compile(r"\n")
_RE_MULTI_SPACE = re.compile(r"\s{2,}")


def clean_for_speech(text: str) -> str:
    """
    Clean text for TTS rendering.

    Removes:
    - Markdown formatting (**, *, #, ```, etc.)
    - Hashtags (#word)
    - URLs
    - Emojis
    - Multiple spaces/newlines

    Converts:
    - Bullet points to spoken equivalents
    - Numbers with context
    """
    if not text:
        return text

    text = _RE_CODE_BLOCK.sub(" code block omitted ", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_HEADER.sub("", text)
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_ITALIC.sub(r"\1", text)
    text = _RE_UNDERLINE_BOLD.sub(r"\1", text)
    text = _RE_UNDERLINE_ITALIC.sub(r"\1", text)
    text = _RE_HASHTAG.sub(r"\1", text)
    text = _RE_URL.sub("", text)
    text = _RE_MD_LINK.sub(r"\1", text)
    text = _RE_EMOJI.sub("", text)
    text = _RE_BULLET.sub("Next, ", text)
    text = _RE_NUMBERED.sub("", text)
    text = _RE_MULTI_NEWLINE.sub(". ", text)
    text = _RE_NEWLINE.sub(" ", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = text.strip()

    if text.endswith("Next,"):
        text = text[:-5].strip()

    return text


def estimate_speech_duration(text: str, wpm: int = 150) -> float:
    """
    Estimate speech duration in seconds.

    Args:
        text: Text to speak
        wpm: Words per minute (default 150 for natural speech)

    Returns:
        Estimated duration in seconds
    """
    word_count = len(text.split())
    return (word_count / wpm) * 60
