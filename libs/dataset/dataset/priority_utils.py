from libs.dataset.dataset.utils import (
    compute_engagement_metric,
    get_issue_metric,
    predict_with_model,
)


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
            100: "high",
            30: "medium",
            5: "low",
        },
    )
    if priority_from_engagement:
        return priority_from_engagement

    # Predict priority using AI if no other methods work
    return predict_with_model(issue, {0: "low", 1: "medium", 2: "high"})
