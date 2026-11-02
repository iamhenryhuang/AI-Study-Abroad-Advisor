import re

# ============================================================
# Page furniture to remove from the text before it goes into web_pages and document_chunks.
#
# Entries are matched as whole words, case-insensitively, so only add phrases that can be
# deleted without changing the meaning of a sentence. A word such as "none" or "true" does not
# belong here: removing it turns "GRE: None required" into "GRE: required".
# Leftover HTML is handled by the tag pattern below, not by listing attribute names.
# ============================================================
KEYWORDS_TO_REMOVE = [
    "Connect with us",
    "Notifications",
    "My Favorites",
    "Settings",
    "Sign Out",
    "Sign In",
    "Committee Information",
    "Go.Grad",
    "SmartRecs",
    "Add funding awards to your favorites list",
    "Get notified of upcoming deadlines and events",
    "Receive personalized recommendations for funding awards",
    "Favorites, recommendations, and notifications are only available for UCLA Graduate Students at this time.",
    "UCLA will never share your email address and you may unsubscribe at any time.",
    "Sign Up",
    "Skip to content",
    "search grad.ucla",
    "Connect with us and learn about our services on the Graduate Education Portal",
    "Contact Us",
    "Twitter",
    "Instagram",
    "YouTube",
    "Skip to main content",
    "Skip to secondary navigation",
    "(link is external)",
    "Skip to Content Top Home About Undergraduate Graduate Partnerships & Research News & Events People Contact Us",
    "Your voice matters! Graduate students—tell us about your experience at U of T.",
    "Your voice matters",
    "Toggle",
    "navigation",
    "submenu",
    "Search",
    "Check your email to complete the gradSERU survey for a chance to win one of seven $3,000 bursaries!",
    "DIVISION OF GRADUATE EDUCATION AND POSTDOCTORAL AFFAIRS UC San Diego",
]

# An HTML tag: "<" directly followed by a tag name. "GPA < 3.0" and "<3.0" are left alone.
_HTML_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")


def remove_keywords(text: str, keywords: list[str]) -> str:
    """Remove leftover HTML tags and every whole-word occurrence of each keyword from text."""
    text = _HTML_TAG.sub(" ", text)
    # Longest first, so "Connect with us and learn..." is not cut short by "Connect with us".
    for kw in sorted(keywords, key=len, reverse=True):
        pattern = rf"(?<!\w){re.escape(kw.strip())}(?!\w)"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Clean up the whitespace left behind, keeping the line breaks that chunking splits on.
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
