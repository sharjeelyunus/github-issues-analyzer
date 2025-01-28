import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class GitHubFetcher:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"  # Request minimal response format
        }

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

    def _fetch_page(self, repo: str, page: int):
        """
        Fetch a single page of issues for a repository.
        """
        url = f"{self.BASE_URL}/repos/{repo}/issues"
        params = {
            "state": "open",
            "per_page": 100,
            "page": page
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 403:  # Rate limit exceeded
            print("Rate limit exceeded. Retrying after delay...")
            time.sleep(10)  # Backoff and retry
            return self._fetch_page(repo, page)
        elif response.status_code != 200:
            raise Exception(f"Failed to fetch issues for {repo}, page {page}: {response.json()}")
        return response.json()

    def fetch_issues(self, repo: str):
        """
        Fetch all open issues for a given repository using parallel pagination.
        """
        # Step 1: Fetch the first page to determine the total number of issues
        first_page_issues = self._fetch_page(repo, 1)
        total_pages = 1
        if len(first_page_issues) == 100:
            # Estimate the total number of pages
            url = f"{self.BASE_URL}/repos/{repo}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                total_issues = response.json().get("open_issues_count", 0)
                total_pages = -(-total_issues // 100)  # Ceiling division
            else:
                print(f"Failed to estimate total pages for {repo}: {response.json()}")

        # Step 2: Fetch remaining pages in parallel
        all_issues = first_page_issues
        if total_pages > 1:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(self._fetch_page, repo, page): page for page in range(2, total_pages + 1)}
                for future in as_completed(futures):
                    try:
                        page_issues = future.result()
                        all_issues.extend(page_issues)
                    except Exception as e:
                        print(f"Failed to fetch page {futures[future]} for {repo}: {e}")

        return all_issues

    def fetch_issues_concurrently(self, repos: list, max_workers=5):
        """
        Fetch issues for multiple repositories concurrently.
        """
        all_issues = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_issues, repo["full_name"]): repo for repo in repos}
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    raw_issues = future.result()
                    print(f"Fetched {len(raw_issues)} issues from {repo['full_name']}")
                    for issue in raw_issues:
                        issue["repo"] = repo["full_name"]
                    all_issues.extend(raw_issues)
                except Exception as e:
                    print(f"Failed to fetch issues for {repo['full_name']}: {e}")
        return all_issues
