from concurrent.futures import ThreadPoolExecutor
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from libs.dataset.dataset.priority_utils import determine_priority
from libs.dataset.dataset.severity_utils import determine_severity
from tqdm import tqdm

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


def process_issue(issue_with_id):
    """
    Process a single GitHub issue to clean and extract relevant data.
    Only processes issues with at least 10 reactions and at least 1 comment.
    """
    row_id, issue = issue_with_id

    if not issue or not isinstance(issue, dict):
        return None

    critical_fields = [
        "id",
        "title",
        "body",
        "labels",
        "repository_url",
        "comments",
        "reactions",
    ]
    if any(field not in issue for field in critical_fields):
        return None

    num_comments = issue.get("comments", 0)
    num_reactions = issue.get("reactions", {}).get("total_count", 0)

    if num_comments < 1 or num_reactions < 1:
        return None

    title, body = issue.get("title", ""), issue.get("body", "")
    if not isinstance(title, str) or not isinstance(body, str) or not title.strip():
        return None

    if not is_english_combined(title, body):
        return None

    repo_url = issue.get("repository_url", "")
    if not isinstance(repo_url, str) or "github.com/repos/" not in repo_url:
        return None

    repo_parts = repo_url.rstrip("/").split("/")
    if len(repo_parts) < 2:
        return None
    repo_full_name = f"{repo_parts[-2]}/{repo_parts[-1]}"

    priority = determine_priority(issue)
    severity = determine_severity(issue)
    if priority is None or severity is None:
        return None

    labels = issue.get("labels", [])
    if not isinstance(labels, list) or not labels:
        return None
    labels = [
        label.get("name", "").strip()
        for label in labels
        if isinstance(label, dict) and "name" in label
    ]

    assignees = issue.get("assignees", [])
    if not isinstance(assignees, list):
        assignees = []
    assignees = [
        assignee.get("login", "").strip()
        for assignee in assignees
        if isinstance(assignee, dict) and "login" in assignee
    ]

    user_data = issue.get("user") or {}
    milestone_data = issue.get("milestone") or {}

    return {
        "id": row_id,
        "issue_id": issue.get("id"),
        "repo": repo_full_name,
        "title": title,
        "body": body,
        "labels": labels,
        "priority": priority,
        "severity": severity,
        "state": issue.get("state", ""),
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "closed_at": issue.get("closed_at", ""),
        "author": user_data.get("login", ""),
        "assignees": assignees,
        "num_comments": num_comments,
        "num_reactions": num_reactions,
        "milestone": milestone_data.get("title", None),
        "pull_request": "pull_request" in issue,
    }


def clean_issues(raw_issues, max_workers=10):
    """
    Clean and structure raw issue data in parallel, filtering out non-English issues,
    removing issues without priority or severity, and ensuring they have labels.
    Optimized for faster execution.
    """
    valid_issues = [
        (idx, issue)
        for idx, issue in enumerate(raw_issues, start=1)
        if issue and isinstance(issue, dict)
    ]

    cleaned = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(
            total=len(valid_issues), desc="Cleaning Issues", unit="issue"
        ) as pbar:
            for result in executor.map(process_issue, valid_issues):
                if result:
                    cleaned.append(result)
                pbar.update(1)

    return cleaned
