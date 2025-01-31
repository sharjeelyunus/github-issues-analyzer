from config import DATASET_REPOS_COUNT, DB_FILE, GITHUB_TOKENS
from libs.dataset.dataset.github_fetcher import GitHubFetcher
from libs.dataset.dataset.issue_cleaner import clean_issues
from libs.dataset.dataset.database_manager import DatabaseManager

def generate_dataset():
    # Initialize GitHubFetcher
    fetcher = GitHubFetcher(GITHUB_TOKENS)

    # Fetch the top repositories globally
    print("Fetching top repositories globally...")
    top_repos = fetcher.fetch_top_repositories(count=DATASET_REPOS_COUNT)

    # Fetch issues concurrently
    print(f"Fetching issues for top {len(top_repos)} repositories concurrently...")
    raw_issues = fetcher.fetch_issues_concurrently(top_repos)

    # Clean and process issues
    print(f"Cleaning {len(raw_issues)} issues...")
    cleaned_issues = clean_issues(raw_issues)

    # Store issues in the SQLite database
    db_manager = DatabaseManager(DB_FILE)
    db_manager.insert_issues(cleaned_issues)
    db_manager.close()
    print(f"Stored {len(cleaned_issues)} issues in the database.")

if __name__ == "__main__":
    generate_dataset()
