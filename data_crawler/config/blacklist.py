# Blacklist for the paths to crawl
IGNORED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mp3", ".avi", ".mov",
    ".css", ".js", ".json", ".xml",
}

BLACKLIST_PATH_FRAGMENTS = [

    # === Authentication / login / SSO ===
    "shibboleth", "log-out", "logout", "/auth/", "/sso/",

    # === Personalization / recommendations / favorites ===
    "smartrec", "my-favorites", "favorites", "building", "oar2",
    # "about" removed: too broad, it would wrongly block valid pages such as about/graduate-admissions

    # === Donations / alumni / funds ===
    # BUG FIX: a comma was missing after "history", so it was joined with the "/news/" below into "history/news/"
    # with the fix, /news/ is filtered correctly
    "giving", "alumni", "endowment", "history",

    # === News / blog / event pages ===
    "/news/", "/events/", "/blog/", "news", "blog",

    # === Campus events / ceremonies ===
    "hooding-ceremony", "commencement", "grad-slam",

    # === Enrollment / academic procedures ===
    "leave-of-absence", "withdrawal", "dissertation",
    "doctoral-committee", "thesis-committee",
    "file-your-thesis", "advancement-to-candidacy",

    # === HR / jobs / faculty positions ===
    "working-at", "postdoc", "for-faculty",
    "human-resources", "payroll", "Faculty", "staff",

    # === Education level (not graduate) ===
    "undergraduate", "/ugrad/", "high-school", "k-12",

    # === Website systems / legal ===
    "sitemap", "privacy-policy", "terms-of-use", "website-feedback",

    # === Colleges / fields (not the target) ===
    "music", "humanities", "architecture", "nursing",
    "public-affairs", "health", "social", "biology", "medicine",
    "law", "school-of-theater-film-and-television",
    "anderson", "david-geffen-school-of-medicine",
    "international-institute", "physical-sciences",
    "chemistry-and-biochemistry-department",
    "school-of-education-and-information-studies",
    "civil-environmental", "dentistry", "accounting",
    "business", "civil", "geography", "environmental", "food", "microbiology", "chemistry", "astronomy",
    "animal", "biotechnology", "arts", "cognitive", "chemical",
    "anthropology", "comparative-literature",

    # === People / community / life ===
    "people", "family", "life", "faculty",

    # === Honors / highlights / promotion ===
    "awards", "spotlight", "spotlights", "insights",

    # === Events / conferences / courses ===
    "seminar", "seminars", "workshop", "workshops", "webinar",
    "conference", "sponsorship",

    # === AI / technology topics ===
    "agentic-ai",

    # === Business / marketing / content ===
    "shirt", "design", "competition", "generating",

    # === Website structure / categories ===
    "taxonomy", "directory", "award",

    # Years
    "2005", "2006", "2007", "2008", "2009",
    "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020",

    # School-specific
    "doctoral", "caltech.edu/academics/courses/", "past-course-offerings",
    "canvas.ucsd.edu", "easy.ucsd.edu","https://cse.umn.edu/cseit/services",
    "https://cse.umn.edu/cseit/self-help-guides","ttps://cse.umn.edu/cs/ta",
    "https://cse.umn.edu/cs/stephen","stories","visit-us","why-bu",

]
