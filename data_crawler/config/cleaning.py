import re

# ============================================================
# Add the phrases you want removed here
# Add any keywords/phrases you want to remove here
# ============================================================
KEYWORDS_TO_REMOVE = [
    "Connect with us",
    "Notifications",
    "My Favorites",
    "Settings",
    "setting",
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
    "Notifications",
    "Contact Us",
    "Twitter",
    "Instagram",
    "YouTube",
    "height",
    "Width",
    "style",
    "none",
    "hidden",
    "visibility",
    "true",
    "iframe",
    "<",">",
    "src",
    "Skip to main content",
    "Skip to secondary navigation",
    "(link is external)",
    "Skip to Content Top Home About Undergraduate Graduate Partnerships & Research News & Events People Contact Us",
    "Your voice matters! Graduate students—tell us about your experience at U of T. ",
    "Your voice matters",
    "Skip to main content <iframe src=\"https://www.googletagmanager.com/ns.html?id=GTM-TZKZB3\" height=\"0\" width=\"0\" style=\"display:none;visibility:hidden\"></iframe>",
    "Toggle",
    "navigation",
    "submenu",
    "Search",
    "Check your email to complete the gradSERU survey for a chance to win one of seven $3,000 bursaries! ",
    "Skip to main  =\"https://www.googletagmanager.com/ns.html?id=GTM-N7QFGH2P\" =\"0\" =\"0\" =\"display:;",
    ":\"/ Graduate Education Office of Graduate and Postdoctoral Education Audience Information For: Prospective Students New Students Current Students Faculty & Staff Main About Tech Program Information Cost & Funding Admissions Professional Development News & Events Quick Actions Apply FAQs Request Info Give ",
    "=\"https://www.googletagmanager.com/ns.html?id=GTM-5LK9HJ2\" =\"0\" =\"0\" =\"display:;:",
    "sk_",
    "DIVISION OF GRADUATE EDUCATION AND POSTDOCTORAL AFFAIRS UC San Diego ",
]

def remove_keywords(text: str, keywords: list[str]) -> str:
    """Remove all occurrences of each keyword from text."""
    for kw in keywords:
        # Use re.escape to handle special characters safely
        pattern = re.escape(kw)
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # Clean up extra whitespace left behind
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()
