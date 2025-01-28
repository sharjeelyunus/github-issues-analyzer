def get_issue_severity(labels):
    """
    Determine the severity of an issue based on its labels.
    """
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
    for label in labels:
        if label.lower() in severity_mapping:
            return severity_mapping[label.lower()]
    return None


def get_severity_from_content(title, body):
    """
    Determine severity based on keywords in the title or body.
    """
    high_severity_keywords = ["crash", "data loss", "security breach", "broken"]
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


def get_issue_severity(labels):
    """
    Determine the severity of an issue based on its labels.
    """
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
    for label in labels:
        if label.lower() in severity_mapping:
            return severity_mapping[label.lower()]
    return None


def get_severity_from_content(title, body):
    """
    Determine severity based on keywords in the title or body.
    """
    high_severity_keywords = ["crash", "data loss", "security breach", "broken"]
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

def get_severity_from_age(created_at, updated_at):
    """
    Determine severity based on issue age and last update time.
    """
    from datetime import datetime
    created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    updated_date = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
    now = datetime.utcnow()

    # If the issue was updated recently, it might be higher severity
    if (now - updated_date).days <= 7:
        return "Critical"

    # If the issue was created within the last 30 days
    if (now - created_date).days <= 30:
        return "Major"

    # If the issue is older, it is less likely to be severe
    return "Minor"

def get_severity_from_engagement(issue):
    """
    Infer severity based on the number of comments and reactions.
    """
    comments = issue.get("comments", 0)
    reactions = issue.get("reactions", {}).get("total_count", 0)
    engagement_score = comments + reactions

    if engagement_score > 50:
        return "Critical"
    elif engagement_score > 10:
        return "Major"
    elif engagement_score > 0:
        return "Minor"
    return None


def determine_severity(issue):
    """
    Determine the severity of an issue using multiple factors.
    """
    labels = [label["name"] for label in issue.get("labels", [])]
    severity_from_labels = get_issue_severity(labels)

    if severity_from_labels:
        return severity_from_labels

    severity_from_content = get_severity_from_content(issue.get("title", ""), issue.get("body", ""))
    if severity_from_content:
        return severity_from_content

    severity_from_engagement = get_severity_from_engagement(issue)
    if severity_from_engagement:
        return severity_from_engagement

    return get_severity_from_age(issue["created_at"], issue["updated_at"])
