import requests
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from itertools import cycle


class GitHubFetcher:
    BASE_URL = "https://api.github.com"

    def __init__(self, tokens: list, max_workers=10, per_repo_workers=5):
        self.tokens = cycle(tokens) 
        self.current_token = next(self.tokens)
        self.headers = {
            "Authorization": f"token {self.current_token}",
            "Accept": "application/vnd.github.v3+json",
        }
        self.max_workers = max_workers
        self.per_repo_workers = per_repo_workers

    def fetch_top_repositories(self, count: int = 10):
        """
        Fetch the top repositories globally by stars, handling pagination efficiently.
        Uses parallel requests for large counts to optimize speed.
        """
        url = f"{self.BASE_URL}/search/repositories"
        max_per_page = 100  # GitHub API limit
        per_page = min(count, max_per_page)
        total_pages = -(-count // per_page)

        repositories = []
        futures = []

        with ThreadPoolExecutor() as executor:
            for page in range(1, total_pages + 1):
                params = {
                    "q": "stars:>0",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                }
                futures.append(
                    executor.submit(
                        requests.get, url, headers=self.headers, params=params
                    )
                )

            for future in as_completed(futures):
                response = future.result()
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch repositories: {response.json()}")

                items = response.json().get("items", [])
                if not items:
                    break

                repositories.extend(items)

        return repositories[:count]

    def _rotate_token(self):
        """Rotate to the next token in the list."""
        self.current_token = next(self.tokens)
        self.headers["Authorization"] = f"token {self.current_token}"

    def _handle_rate_limit(self):
        """Wait until the rate limit resets."""
        print("⏳ All tokens exhausted! Sleeping for 60 seconds...")
        time.sleep(60)

    def _fetch_page(self, repo: str, page: int, retries=3):
        """
        Fetch a single page of issues for a repository with retries and token rotation.
        """
        one_year_ago = (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z"
        url = f"{self.BASE_URL}/repos/{repo}/issues"
        params = {
            "state": "all",
            "since": one_year_ago,
            "per_page": 100,
            "page": page,
        }

        for attempt in range(retries):
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code == 200:
                return response.json()

            if response.status_code == 403:  # Rate limit exceeded
                print(f"⚠️ Rate limit exceeded for token. Rotating token...")
                self._rotate_token()  # Switch to next token
                continue

            elif response.status_code == 401:  # Invalid token
                print(f"❌ Invalid token detected, rotating...")
                self._rotate_token()
                continue

            else:
                print(
                    f"❌ Failed to fetch {repo} (Page {page}): {response.status_code}"
                )
                return []

        self._handle_rate_limit()  # If all tokens fail, sleep
        return self._fetch_page(repo, page)  # Retry after sleeping

    def _fetch_repo_issues_parallel(self, repo: str):
        """
        Fetch all issues for a repository using parallel page requests.
        """
        all_issues = []
        page = 1

        first_page = self._fetch_page(repo, page)
        if not first_page:
            return []

        all_issues.extend(first_page)

        with ThreadPoolExecutor(max_workers=self.per_repo_workers) as executor:
            futures = {}
            while True:
                page += 1
                futures[executor.submit(self._fetch_page, repo, page)] = page
                for future in as_completed(futures):
                    page_issues = future.result()
                    if not page_issues:
                        return all_issues
                    all_issues.extend(page_issues)

        return all_issues

    def fetch_issues_concurrently(self, repos: list):
        """
        Fetch issues for multiple repositories concurrently with progress tracking.
        """
        all_issues = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor, tqdm(
            total=len(repos), desc="Fetching issues", unit="repo"
        ) as pbar:
            futures = {
                executor.submit(
                    self._fetch_repo_issues_parallel, repo["full_name"]
                ): repo
                for repo in repos
            }

            for future in as_completed(futures):
                repo = futures[future]
                try:
                    repo_issues = future.result()
                    all_issues.extend(repo_issues)
                except Exception as e:
                    print(f"❌ Failed to fetch issues for {repo['full_name']}: {e}")
                pbar.update(1)

        print(f"📊 Total issues fetched: {len(all_issues)}")
        return all_issues
