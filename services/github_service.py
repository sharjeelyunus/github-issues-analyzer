import requests
from config import GITHUB_TOKEN, REPO_OWNER, REPO_NAME

def fetch_github_issues(state="open", per_page=30, repo_owner=REPO_OWNER, repo_name=REPO_NAME):
    """Fetch issues from the GitHub API, including their existing labels."""
    page = 1
    issues = []

    while True:
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
        params = {"state": state, "page": page, "per_page": per_page}

        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"GitHub API error: {response.status_code} {response.text}")

        data = response.json()
        if not data:
            break

        for issue in data:
            if "pull_request" not in issue:
                issues.append({
                    "id": issue.get("id"),
                    "number": issue.get("number"),
                    "title": issue.get("title"),
                    "body": issue.get("body"),
                    "labels": [label["name"] for label in issue.get("labels", [])]
                })
        page += 1

    return issues

def fetch_repo_labels():
    """Fetch labels for a repository."""
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/labels"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code} {response.text}")

    labels = []
    for label in response.json():
        name = label['name']
        description = label.get('description', f"This issue is related to {name}")
        labels.append({"name": name, "description": description})

    return labels

def fetch_top_repositories():
    """Fetch the top 5 repositories from GitHub."""
    url = "https://api.github.com/search/repositories?q=stars:>50000&sort=stars&order=desc&per_page=5"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    response = requests.get(url, headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch top repositories: {response.status_code}, {response.text}")
    repos = response.json()["items"]
    return [repo["full_name"] for repo in repos]
