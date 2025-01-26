import sqlite3
from fastapi import FastAPI

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DB_FILE = "issues.db"

app = FastAPI(
    title="GitHub Issues API",
    description="Fetch analyzed GitHub issues, duplicates, and labels from the local database.",
    version="1.1.0"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def fetch_all_issues():
    """Fetch all issues from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT github_id, title, body, duplicates, labels FROM issues")
    issues = cursor.fetchall()
    conn.close()

    result = []
    for github_id, title, body, duplicates, labels in issues:
        duplicate_list = eval(duplicates)
        label_list = eval(labels) if labels else []
        for duplicate in duplicate_list:
            duplicate["similarity"] = round(duplicate["similarity"] * 100, 2)

        result.append({
            "github_id": github_id,
            "title": title,
            "body": body,
            "duplicates": duplicate_list,
            "labels": label_list,
        })
    return result


def fetch_issue_by_id(github_id: int):
    """Fetch a specific issue by GitHub ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT github_id, title, body, duplicates, labels FROM issues WHERE github_id = ?", (github_id,))
    issue = cursor.fetchone()
    conn.close()

    if issue:
        duplicate_list = eval(issue[3])
        label_list = eval(issue[4]) if issue[4] else []
        for duplicate in duplicate_list:
            duplicate["similarity"] = round(duplicate["similarity"] * 100, 2)

        return {
            "github_id": issue[0],
            "title": issue[1],
            "body": issue[2],
            "duplicates": duplicate_list,
            "labels": label_list,
        }
    return None


def count_duplicates(issues):
    """Count issues that have potential duplicates."""
    return sum(1 for issue in issues if issue["duplicates"])


# -----------------------------------------------------------------------------
# API ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/")
def read_root():
    """
    Root endpoint for the API.
    """
    return {
        "message": "Welcome to the GitHub Issues Analyzer!",
        "endpoints": {
            "/issues": "List all issues.",
            "/issues/{github_id}": "Get details of a specific issue.",
            "/duplicates": "List issues with potential duplicates.",
            "/labels": "List all labels for issues."
        },
    }


@app.get("/issues")
def get_all_issues():
    """Return all issues along with metadata."""
    issues = fetch_all_issues()
    return {
        "total": len(issues),
        "duplicates_count": count_duplicates(issues),
        "labeled_issues_count": sum(1 for issue in issues if issue["labels"]),
        "issues": issues,
    }

@app.get("/issues/{github_id}")
def get_issue(github_id: int):
    """Return a single issue by GitHub ID."""
    issue = fetch_issue_by_id(github_id)
    if issue:
        return issue
    return {"error": "Issue not found"}


@app.get("/duplicates")
def get_duplicates():
    """Return issues that have potential duplicates along with metadata."""
    all_issues = fetch_all_issues()
    duplicates = [issue for issue in all_issues if issue["duplicates"]]
    return {
        "total": len(all_issues),
        "duplicates_count": len(duplicates),
        "labeled_issues_count": sum(1 for issue in duplicates if issue["labels"]),
        "issues": duplicates,
    }


@app.get("/labels")
def get_labels():
    """Return all issues with their labels."""
    issues = fetch_all_issues()
    labeled_issues = [issue for issue in issues if issue["labels"]]
    return {
        "total": len(issues),
        "labeled_issues_count": len(labeled_issues),
        "issues": labeled_issues,
    }
