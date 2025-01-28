from concurrent.futures import ThreadPoolExecutor
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from libs.dataset.dataset.priority_utils import determine_priority
from libs.dataset.dataset.severity_utils import determine_severity

DetectorFactory.seed = 0


def is_english_combined(title: str, body: str) -> bool:
    """
    Check if the combined text of title and body is in English.
    """
    combined_text = f"{title} {body}".strip()
    if not combined_text:
        return False
    try:
        return detect(combined_text) == "en"
    except LangDetectException:
        return False


def process_issue(issue):
    """
    Process a single issue to clean and extract relevant data.
    """
    if "pull_request" in issue or not issue.get("labels"):
        return None

    title = issue.get("title", "")
    body = issue.get("body", "")

    if not is_english_combined(title, body):
        return None

    # Compute priority and severity
    priority = determine_priority(issue)
    severity = determine_severity(issue)

    return {
        "id": issue.get("id"),
        "repo": issue.get("repository_url", "").split("/")[-1],
        "title": title,
        "body": body,
        "labels": [label["name"] for label in issue.get("labels", [])],
        "priority": priority,
        "severity": severity,
    }


def clean_issues(raw_issues, max_workers=10):
    """
    Clean and structure raw issue data in parallel, filtering out non-English issues.
    """
    cleaned = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_issue, issue) for issue in raw_issues]
        for future in futures:
            try:
                result = future.result()
                if result:
                    cleaned.append(result)
            except Exception as e:
                print(f"Error processing issue: {e}")
    return cleaned
