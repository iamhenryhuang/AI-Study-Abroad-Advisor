"""Crawler configuration: tuning parameters, ignored file types, per-school root URLs and the
keyword list used to clean page text."""
from .parameter import (
    CONFIG,
    USER_AGENT,
    URL_PATH_HINTS,
)
from .blacklist import IGNORED_EXTENSIONS
from .root_url import SCHOOLS
from .cleaning import KEYWORDS_TO_REMOVE, remove_keywords

__all__ = [
    "CONFIG",
    "USER_AGENT",
    "URL_PATH_HINTS",
    "IGNORED_EXTENSIONS",
    "SCHOOLS",
    "KEYWORDS_TO_REMOVE",
    "remove_keywords",
]
