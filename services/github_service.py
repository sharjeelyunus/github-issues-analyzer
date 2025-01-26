import requests
from config import GITHUB_TOKEN, REPO_OWNER, REPO_NAME

def fetch_github_issues(state="open", per_page=30):
    """Fetch issues from the GitHub API."""
    page = 1
    issues = []

    while True:
        url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues"
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
        params = {"state": state, "page": page, "per_page": per_page}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"GitHub API error: {response.status_code} {response.text}")

        data = response.json()
        if not data:
            break
        issues.extend(issue for issue in data if "pull_request" not in issue)
        page += 1

    return issues
