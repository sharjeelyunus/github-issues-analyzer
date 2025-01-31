from libs.dataset.dataset.utils import compute_engagement_metric, get_issue_metric
import re


def get_priority_from_content(title, body):
    """
    Determine priority based on keywords in the title or body.
    Uses case-insensitive search and word boundaries for better matching.
    """
    high_priority_keywords = [
        "urgent",
        "critical",
        "immediately",
        "p0",
        "high priority",
        "severe",
        "must fix",
        "showstopper",
        "blocker",
        "breaking",
        "down",
        "does not work",
        "fails completely",
        "security risk",
        "unusable",
    ]
    medium_priority_keywords = [
        "moderate",
        "p1",
        "medium priority",
        "degraded performance",
        "intermittent failure",
        "affects usability",
        "causes issues",
        "affects multiple users",
    ]
    low_priority_keywords = [
        "low priority",
        "p2",
        "nice to have",
        "cosmetic issue",
        "minor inconvenience",
        "does not affect functionality",
        "small glitch",
        "aesthetic",
        "trivial",
    ]

    content = f"{title} {body}".lower()

    if any(
        re.search(rf"\b{re.escape(keyword)}\b", content)
        for keyword in high_priority_keywords
    ):
        return "High"
    elif any(
        re.search(rf"\b{re.escape(keyword)}\b", content)
        for keyword in medium_priority_keywords
    ):
        return "Medium"
    elif any(
        re.search(rf"\b{re.escape(keyword)}\b", content)
        for keyword in low_priority_keywords
    ):
        return "Low"

    return None


def determine_priority(issue):
    """
    Determine the priority of an issue using multiple factors.
    """
    labels = [label["name"] for label in issue.get("labels", [])]
    priority_mapping = {
        "priority: high": "high",
        "priority: medium": "medium",
        "priority: low": "low",
        "p0": "high",
        "p1": "medium",
        "p2": "low",
        "critical": "high",
        "urgent": "high",
    }
    # Determine priority from labels
    priority_from_labels = get_issue_metric(labels, priority_mapping)
    if priority_from_labels:
        return priority_from_labels

    # Determine priority from engagement
    priority_from_engagement = compute_engagement_metric(
        issue,
        {
            50: "high",
            10: "medium",
            0: "low",
        },
    )
    if priority_from_engagement:
        return priority_from_engagement

    if issue.get("title") or issue.get("body"):
        # Determine priority from content
        priority_from_content = get_priority_from_content(
            issue.get("title", ""), issue.get("body", "")
        )
        if priority_from_content:
            return priority_from_content

    return None
