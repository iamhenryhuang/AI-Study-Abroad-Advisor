"""School alias table and school detection.

This module has no dependencies, so the agent, the search backends and the db loaders can all use it
without importing the LangGraph graph.

To add a school, add it here and to data_crawler/config/root_url.py if it should also be crawled. The
table may list more schools than the crawler does, because community reports cover more schools.
"""
from __future__ import annotations

import re

SCHOOL_ALIASES: dict[str, list[str]] = {
    "cmu":          ["cmu", "carnegie mellon", "卡內基梅隆"],
    "mit":          ["mit", "massachusetts institute", "麻省理工"],
    "stanford":     ["stanford", "史丹佛", "斯坦福"],
    "caltech":      ["caltech", "california institute", "加州理工"],
    "gatech":       ["georgia tech", "gatech", "georgia institute of technology", "喬治亞理工"],
    "ucla":         ["ucla", "uc los angeles", "加州洛杉磯"],
    "ucsd":         ["ucsd", "uc san diego", "加州聖地牙哥"],
    "umass":        ["umass", "amherst", "麻州大學"],
    "berkeley":     ["berkeley", "uc berkeley", "柏克萊"],
    "washington":   ["university of washington", "uw", "華盛頓大學"],
    "washu":        ["washu", "wustl", "washington university in st", "聖路易華盛頓"],
    "uiuc":         ["uiuc", "illinois", "urbana-champaign", "伊利諾"],
    "cornell":      ["cornell", "康乃爾"],
    "princeton":    ["princeton", "普林斯頓"],
    "columbia":     ["columbia", "哥倫比亞"],
    "upenn":        ["upenn", "university of pennsylvania", "penn", "賓州大學", "賓夕法尼亞"],
    "umich":        ["umich", "michigan", "密西根"],
    "umn":          ["umn", "university of minnesota", "明尼蘇達"],
    "utaustin":     ["ut austin", "university of texas at austin", "texas austin", "德州大學奧斯汀"],
    "utoronto":     ["utoronto", "university of toronto", "uoft", "多倫多大學"],
    "ucsb":         ["ucsb", "santa barbara", "聖塔芭芭拉"],
    "uci":          ["uci", "uc irvine", "irvine", "爾灣"],
    "purdue":       ["purdue", "普渡"],
    "wisc":         ["wisconsin", "madison", "威斯康辛"],
    "umd":          ["umd", "maryland", "college park", "馬里蘭"],
    "usc":          ["usc", "university of southern california", "南加大", "南加州大學"],
    "nyu":          ["nyu", "new york university", "紐約大學"],
    "ucsc":         ["ucsc", "santa cruz", "聖塔克魯茲"],
    "rice":         ["rice", "萊斯大學"],
    "northeastern": ["northeastern", "東北大學"],
    "ucdavis":      ["ucdavis", "uc davis", "戴維斯"],
    "osu":          ["ohio state", "osu", "俄亥俄州立"],
    "duke":         ["duke", "杜克"],
    "rutgers":      ["rutgers", "羅格斯"],
    "bu":           ["bu", "boston university", "波士頓大學"],
}

# Different schools whose names contain an alias above. They are blanked out before matching, so
# "University of British Columbia" is not filed under Columbia.
_OTHER_SCHOOLS = re.compile(
    r"british columbia|columbia college"
    r"|penn state|pennsylvania state"
    r"|michigan state|michigan tech"
    r"|illinois institute|illinois tech|illinois state|illinois chicago"
    r"|(?:northern|southern|eastern|western|northeastern) illinois"
    r"|james madison"
    r"|umass (?:boston|lowell|dartmouth)|amherst college"
)


def _alias_pattern(alias: str) -> re.Pattern:
    """Latin aliases must match whole words: a substring match finds "mit" in "admitted" and "submit",
    and "rice" in "price". Chinese has no word boundaries, so those aliases match as substrings."""
    if alias.isascii():
        return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
    return re.compile(re.escape(alias))


_ALIAS_PATTERNS = {
    school_id: [_alias_pattern(alias) for alias in aliases]
    for school_id, aliases in SCHOOL_ALIASES.items()
}


def detect_school_ids(text: str) -> list[str]:
    """Return the school ids mentioned in a text, in the order they first appear."""
    low = _OTHER_SCHOOLS.sub(" ", (text or "").lower())
    first_seen: dict[str, int] = {}
    for school_id, patterns in _ALIAS_PATTERNS.items():
        positions = [m.start() for m in (p.search(low) for p in patterns) if m]
        if positions:
            first_seen[school_id] = min(positions)
    return sorted(first_seen, key=first_seen.get)
