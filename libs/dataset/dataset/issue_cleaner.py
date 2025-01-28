from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from libs.dataset.dataset.priority_utils import determine_priority
from libs.dataset.dataset.severity_utils import determine_severity

DetectorFactory.seed = 0

def is_english(text: str) -> bool:
    """
    Check if the given text is in English using langdetect.
    """
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

def clean_issues(raw_issues):
    """
    Clean and structure raw issue data, filtering out non-English issues and ensuring the required structure.
    """
    cleaned = []
    for issue in raw_issues:
        if "pull_request" not in issue and issue.get("labels"):
            title = issue.get("title", "")
            body = issue.get("body", "")
            if is_english(title) and (not body or is_english(body)):
                priority = determine_priority(issue)
                severity = determine_severity(issue)
                cleaned.append({
                    "id": issue.get("id"),
                    "repo": issue.get("repository_url", "").split("/")[-1],
                    "title": title,
                    "body": body,
                    "labels": [label["name"] for label in issue.get("labels", [])],
                    "priority": priority,
                    "severity": severity,
                })
    return cleaned
