# File types the crawler never follows. Whether a page is about admissions is decided by the rules in
# url_tools.application_scope_exclusion and by the LLM URL filter, not by a path blacklist.
IGNORED_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".mp3", ".avi", ".mov",
    ".css", ".js", ".json", ".xml",
}
