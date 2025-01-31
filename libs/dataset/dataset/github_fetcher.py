import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
from tqdm import tqdm  # Progress bar


class GitHubFetcher:
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str, max_workers=10):
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.max_workers = max_workers

    def fetch_top_repositories(self, count: int = 10):
        """
        Fetch the top repositories globally by stars.
        """
        url = f"{self.BASE_URL}/search/repositories"
        params = {
            "q": "stars:>0",
            "sort": "stars",
            "order": "desc",
            "per_page": count,
        }
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repositories: {response.json()}")
        return response.json().get("items", [])

    def _fetch_page(self, repo: str, page: int):
        """
        Fetch a single page of issues for a repository.
        """
        one_year_ago = (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z"
        url = f"{self.BASE_URL}/repos/{repo}/issues"
        params = {
            "state": "all",
            "since": one_year_ago,
            "per_page": 100,
            "page": page,
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 403:
            print("⚠️ Rate limit exceeded. Retrying in 60 seconds...")
            time.sleep(60)
            return self._fetch_page(repo, page)
        elif response.status_code != 200:
            return []

        return response.json()

    def fetch_all_pages(self, repo: str):
        """
        Fetch all pages of issues for a repository in parallel.
        """
        all_issues = []
        page = 1

        while True:
            page_issues = self._fetch_page(repo, page)
            if not page_issues:
                break
            all_issues.extend(page_issues)
            page += 1

        return all_issues

    def fetch_issues_concurrently(self, repos: list):
        """
        Fetch all issues for multiple repositories concurrently with a loading indicator.
        """
        all_issues = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor, tqdm(
            total=len(repos), desc="Fetching issues", unit="repo"
        ) as pbar:
            futures = {
                executor.submit(self.fetch_all_pages, repo["full_name"]): repo
                for repo in repos
            }
            for future in as_completed(futures):
                repo = futures[future]
                try:
                    raw_issues = future.result()
                    all_issues.extend(raw_issues)
                except Exception as e:
                    print(f"❌ Failed to fetch issues for {repo['full_name']}: {e}")
                pbar.update(1)

        print(f"📊 Total issues fetched across all repositories: {len(all_issues)}")
        return all_issues
