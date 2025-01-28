def get_issue_priority(labels):
    """
    Determine issue priority based on labels.
    """
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

    for label in labels:
        normalized_label = label.lower()
        if normalized_label in priority_mapping:
            return priority_mapping[normalized_label]
    return "unknown"

def get_engagement_priority(issue):
    """
    Determine priority based on comments and reactions.
    """
    comments = issue.get("comments", 0)
    reactions = issue.get("reactions", {}).get("total_count", 0)
    engagement_score = comments + reactions

    if engagement_score > 50:
        return "high"
    elif engagement_score > 10:
        return "medium"
    else:
        return "low"


def is_recent_issue(created_at, updated_at):
    """
    Determine priority based on creation and last update time.
    """
    from datetime import datetime
    try:
        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        updated_date = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
        now = datetime.utcnow()

        if (now - updated_date).days <= 7:
            return "high"
        elif (now - created_date).days <= 30:
            return "medium"
        else:
            return "low"
    except Exception as e:
        print(f"Error parsing dates for issue: {e}")
        return "unknown"


def determine_priority(issue):
    """
    Determine the priority of an issue using multiple factors.
    """
    labels = [label["name"] for label in issue.get("labels", [])]

    priority_from_labels = get_issue_priority(labels)

    if priority_from_labels != "unknown":
        return priority_from_labels

    priority_from_engagement = get_engagement_priority(issue)
    if priority_from_engagement == "high":
        return "high"

    return is_recent_issue(issue["created_at"], issue["updated_at"])
