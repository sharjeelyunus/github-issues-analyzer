import sqlite3
from fastapi import FastAPI

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DB_FILE = "issues.db"

app = FastAPI(
    title="GitHub Issues API",
    description="Fetch analyzed GitHub issues and duplicates from the local database.",
    version="1.0.0"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def fetch_all_issues():
    """Fetch all issues from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT github_id, title, body, duplicates FROM issues")
    issues = cursor.fetchall()
    conn.close()

    result = []
    for github_id, title, body, duplicates in issues:
        duplicate_list = eval(duplicates)
        for duplicate in duplicate_list:
            duplicate["similarity"] = round(duplicate["similarity"] * 100, 2)

        result.append({
            "github_id": github_id,
            "title": title,
            "body": body,
            "duplicates": duplicate_list,
        })
    return result


def fetch_issue_by_id(github_id: int):
    """Fetch a specific issue by GitHub ID."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT github_id, title, body, duplicates FROM issues WHERE github_id = ?", (github_id,))
    issue = cursor.fetchone()
    conn.close()

    if issue:
        duplicate_list = eval(issue[3])
        for duplicate in duplicate_list:
            duplicate["similarity"] = round(duplicate["similarity"] * 100, 2)

        return {
            "github_id": issue[0],
            "title": issue[1],
            "body": issue[2],
            "duplicates": duplicate_list,
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
            "/duplicates": "List issues with potential duplicates."
        },
    }


@app.get("/issues")
def get_all_issues():
    """Return all issues along with metadata."""
    issues = fetch_all_issues()
    return {
        "total": len(issues),
        "duplicates": count_duplicates(issues),
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
        "duplicates": len(duplicates),
        "issues": duplicates,
    }
