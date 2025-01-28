import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

class GitHubFetcher:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str):
        self.headers = {"Authorization": f"token {token}"}

    def fetch_top_repositories(self, count: int = 10):
        """
        Fetch the top repositories globally by stars.
        """
        url = f"{self.BASE_URL}/search/repositories"
        params = {
            "q": "stars:>0",
            "sort": "stars",
            "order": "desc",
            "per_page": count
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repositories: {response.json()}")
        return response.json().get("items", [])

    def fetch_issues(self, repo: str):
        """
        Fetch all open issues for a given repository using pagination.
        """
        all_issues = []
        page = 1
        while True:
            url = f"{self.BASE_URL}/repos/{repo}/issues"
            params = {
                "state": "open",
                "per_page": 100,
                "page": page
            }
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch issues: {response.json()}")
            issues = response.json()
            if not issues:
                break
            all_issues.extend(issues)
            page += 1
        return all_issues

    def fetch_all_repos_issues(self, repos: list):
        """
        Fetch issues for multiple repositories concurrently.
        """
        all_issues = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_repo = {executor.submit(self.fetch_issues, repo["full_name"]): repo for repo in repos}
            for future in as_completed(future_to_repo):
                repo = future_to_repo[future]
                try:
                    issues = future.result()
                    for issue in issues:
                        issue["repo"] = repo["full_name"]
                    all_issues.extend(issues)
                except Exception as exc:
                    print(f"Failed to fetch issues for {repo['full_name']}: {exc}")
        return all_issues
