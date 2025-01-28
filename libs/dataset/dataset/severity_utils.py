from libs.dataset.dataset.utils import (
    compute_engagement_metric,
    get_issue_metric,
    predict_with_model,
)


def get_severity_from_content(title, body):
    """
    Determine severity based on keywords in the title or body.
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
    ]
    medium_severity_keywords = ["performance", "slow", "delay"]
    low_severity_keywords = ["typo", "minor", "cosmetic", "UI improvement"]

    content = f"{title} {body}".lower()
    if any(keyword in content for keyword in high_severity_keywords):
        return "Critical"
    elif any(keyword in content for keyword in medium_severity_keywords):
        return "Major"
    elif any(keyword in content for keyword in low_severity_keywords):
        return "Minor"
    return None


def determine_severity(issue):
    """
    Determine the severity of an issue using multiple factors.
    """
    labels = [label["name"] for label in issue.get("labels", [])]
    severity_mapping = {
        "severity: critical": "Critical",
        "critical": "Critical",
        "blocker": "Critical",
        "severity: major": "Major",
        "major": "Major",
        "severity: minor": "Minor",
        "minor": "Minor",
        "trivial": "Minor",
    }
    severity_from_labels = get_issue_metric(labels, severity_mapping)

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

    return predict_with_model(issue, {0: "Minor", 1: "Major", 2: "Critical"})
