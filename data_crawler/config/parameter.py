from dataclasses import dataclass
# Crawler parameters

# ===================== #
# url crawler parameter #
# ===================== #
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

@dataclass(frozen=True)
class CrawlConfig:
    MAX_DEPTH: int = 5
    CRAWL_DELAY_MS: int = 400
    RENDER_WAIT_MS: int = 800
    STRIP_QUERY: bool = True
    MAX_PAGES: int = 800
    NUM_WORKERS: int = 5
CONFIG = CrawlConfig()


# Reference scores for the LLM page classifier. Matched as substrings of the lowercased URL path.
URL_PATH_HINTS = {
    "admissions": [
        ("admissions", 6), ("admission", 3.5), ("apply", 5),
        ("application", 4), ("how-to-apply", 5), ("how_to_apply", 5),
        ("checklist", 4), ("requirements", 5), ("eligibility", 10),
        ("program", 3), ("undergraduate", -5), ("phd", -3), ("prospective", 3),
        ("english-language-proficiency", 8), ("proficiency", 4),
        ("international-applicants", 8), ("guidelines", 4), ("international", 8),
        ("check", 5), ("ready-to-apply", 6), ("fellowships", 4),
        ("tips-for-applying", 5), ("international/ready-to-apply", 10),
        ("international/before-to-apply", 10), ("domestic/before-to-apply", 10),
        ("/domestic/ready-you-apply", 10), ("step", 4), ("process", 6),
        ("timelines", 8), ("english-proficiency", 8), ("gpa", 15), ("gre", 10),
    ],
    "program": [
        ("grad_cs", 5), ("graduate", 5), ("ms-program", 5),
        ("ms_program", 5), ("degree", 8), ("curriculum", 4),
        ("degree-programs", 5), ("degree_programs", 5), ("undergraduate", -5),
        ("program", 4), ("computer-science", 3),
    ],
    "tuition": [
        ("tuition", 10), ("financial", 10), ("financial-assistance", 5),
        ("financial_assistance", 5), ("cost", 10), ("fees", 10),
        ("scholarship", 15), ("fellowships", 15), ("funding", 4),
        ("reimbursement", 6), ("tax", 5),
    ],
    "faq": [
        ("faq", 8), ("faqs", 8), ("faq-applicants", 6),
        ("frequently-asked", 7), ("frequently_asked", 6),
    ],
}
