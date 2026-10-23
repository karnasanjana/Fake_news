"""Removes source giveaways from ISOT articles, shared by training and the app.

Almost every real article starts with a "WASHINGTON (Reuters) -" dateline, and many fake
ones carry "21st Century Wire", Twitter handles or "Featured image via ...". A model can
hit ~99% just by spotting those, so they are stripped before training and prediction.
"""
import re

LEAKY_PATTERNS = [
    r"^.{0,150}?\(reuters\)\s*-\s*",          # "WASHINGTON (Reuters) - " dateline
    r"\breuters\b",
    r"21st\s*century\s*wire\w*|\b21wire\b",
    r"featured image via.*$",
    r"(photo|image) (by|via|credit)[^.]*",
    r"getty images",
    r"pic\.twitter\.com/\S+",
    r"https?://\S+|www\.\S+",
    r"@\w+",
]
_leaky = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in LEAKY_PATTERNS]


def strip_source_leakage(text):
    for pattern in _leaky:
        text = pattern.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
