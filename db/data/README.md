# db/data

This folder is empty on purpose. The loaders and the retrieval code read these files from here, and nothing in `db/data/*.json` is committed.

School information is crawled by `data_crawler` and does not come from this folder. The files below are different: community reports and faculty are collected by hand. In our project some of them were built manually, partly because the sites block automated access and partly for convenience. The system only depends on the format below, not on how a file was produced, so you can fill them by hand, by script, or from your own sources.

| File | Used by | Required for |
|---|---|---|
| `professors.json` | `python backend/run.py load-professors` | the "which professors work on X" question |
| `gradcafe_results.json` | `python db/loaders/load_applicant_reports.py` | Experience retrieval |
| `1point3.json` | `python db/loaders/load_applicant_reports.py` | Experience retrieval |
| `1point3_distribution.json` | `backend/retriever/recommend.py` | school recommendation |

## professors.json

A list of faculty records. Each `school_id` must already exist in `universities`, so crawl that school first.

```json
[
  {
    "school_id": "stanford",
    "name": "Jane Doe",
    "title": "Associate Professor",
    "research_areas": ["Machine Learning", "Natural Language Processing"],
    "profile_url": "https://example.edu/people/jane-doe"
  }
]
```

Rows are upserted on `(university_id, name)`. Collect names, titles and research areas from each department's public people page.

## gradcafe_results.json

A list of GradCafe result rows. The loader reads `id`, `school`, `program`, `level` (`Masters`, `PhD`, ...), `decision` (`Accepted`, `Rejected`, `Wait listed`, `Interview`, `Other`), `ugpa`, `greq`, `grev`, `grew`, `status`, `season` and `notes`. Rows without a `school` are skipped. Whatever way you collect them, respect GradCafe's terms of use.

## 1point3.json

An object keyed by school. Each value is a list of posts with `url`, `school`, `title`, `content`, `major`, `undergrad_school`, `gpa`, `toefl`, `gre` and `result`. Whatever way you collect them, respect the forum's terms of use.

## 1point3_distribution.json

Per-school medians used to tier schools as reach, match or safety:

```json
{
  "university_admissions_data": {
    "Stanford University": {
      "school_id": "stanford",
      "median_gpa": 3.9,
      "median_toefl": 108,
      "median_gre": 330
    }
  }
}
```

The numbers above are placeholders to show the shape, not real statistics.
