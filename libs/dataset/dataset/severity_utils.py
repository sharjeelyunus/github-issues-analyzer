from libs.dataset.dataset.utils import compute_engagement_metric, get_fuzzy_metric, get_issue_metric
import re


def get_severity_from_content(title, body):
    """
    Determine severity based on keywords in the title or body.
    Uses case-insensitive search and word boundaries for better matching.
    """
    high_severity_keywords = [
        "crash",
        "data loss",
        "security breach",
        "broken",
        "bug",
        "error",
        "failure",
        "failed",
        "corrupt",
        "freeze",
        "unstable",
    ]
    medium_severity_keywords = [
        "performance",
        "slow",
        "delay",
        "lag",
        "high latency",
        "memory leak",
    ]
    low_severity_keywords = [
        "typo",
        "minor",
        "cosmetic",
        "UI improvement",
        "grammar",
        "misspelled",
    ]

    content = f"{title} {body}".lower()

    if any(
        re.search(rf"\b{re.escape(keyword)}\b", content)
        for keyword in high_severity_keywords
    ):
        return "Critical"
    elif any(
        re.search(rf"\b{re.escape(keyword)}\b", content)
        for keyword in medium_severity_keywords
    ):
        return "Major"
    elif any(
        re.search(rf"\b{re.escape(keyword)}\b", content)
        for keyword in low_severity_keywords
    ):
        return "Minor"

    return None


def determine_severity(issue):
    """
    Determine the severity of an issue using multiple factors.
    """
    labels = [label["name"] for label in issue.get("labels", [])]
    severity_keywords = {
        "Critical": ["severity: critical", "critical", "blocker", "sev1"],
        "Major": ["severity: major", "major", "sev2"],
        "Minor": ["severity: minor", "minor", "trivial", "sev3"]
    }

    severity_from_labels = get_fuzzy_metric(labels, severity_keywords)

    if severity_from_labels:
        return severity_from_labels

    severity_from_content = get_severity_from_content(
        issue.get("title", ""), issue.get("body", "")
    )
    if severity_from_content:
        return severity_from_content

    severity_from_engagement = compute_engagement_metric(
        issue,
        {
            50: "Critical",
            10: "Major",
            0: "Minor",
        },
    )
    if severity_from_engagement:
        return severity_from_engagement

    return None
