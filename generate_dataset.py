from config import GITHUB_TOKEN
from libs.dataset.dataset.github_fetcher import GitHubFetcher
from libs.dataset.dataset.issue_cleaner import clean_issues
from libs.dataset.dataset.database_manager import DatabaseManager

def main():
    # Initialize GitHubFetcher
    fetcher = GitHubFetcher(GITHUB_TOKEN)

    # Fetch the top repositories globally
    print("Fetching top repositories globally...")
    top_repos = fetcher.fetch_top_repositories(count=20)

    # Fetch and clean issues for each repository
    all_issues = []
    for repo in top_repos:
        repo_name = repo["full_name"]
        print(f"Fetching issues for {repo_name}...")
        raw_issues = fetcher.fetch_issues(repo_name)
        cleaned_issues = clean_issues(raw_issues)
        for issue in cleaned_issues:
            issue["repo"] = repo_name
        all_issues.extend(cleaned_issues)

    # Store issues in the SQLite database
    db_manager = DatabaseManager("github_issues.db")
    db_manager.insert_issues(all_issues)
    print(f"Stored {len(all_issues)} issues in the SQLite database.")

if __name__ == "__main__":
    main()
