import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DATASET_REPOS_COUNT, DB_FILE, GITHUB_TOKENS
from libs.dataset.dataset.github_fetcher import GitHubFetcher
from libs.dataset.dataset.issue_cleaner import clean_issues
from libs.dataset.dataset.database_manager import DatabaseManager


def generate_dataset():
    start_time = time.time()

    fetcher = GitHubFetcher(GITHUB_TOKENS, max_workers=20, per_repo_workers=8)

    # Fetch the top repositories
    print("Fetching top repositories...")
    top_repos = fetcher.fetch_top_repositories(count=DATASET_REPOS_COUNT)
    print(f"✅ Fetched {len(top_repos)} repositories.")

    # Fetch issues concurrently
    print(f"Fetching issues for {len(top_repos)} repositories concurrently...")
    raw_issues = fetcher.fetch_issues_concurrently(top_repos)
    print(f"✅ Fetched {len(raw_issues)} raw issues.")

    print(f"Cleaning and inserting {len(raw_issues)} issues...")

    # Open the database connection
    db_manager = DatabaseManager(DB_FILE)
    db_manager.open_connection()

    batch_size = 20000
    cleaned_count = 0

    try:
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(clean_issues, raw_issues[i : i + batch_size]): i
                for i in range(0, len(raw_issues), batch_size)
            }

            for future in as_completed(futures):
                batch_index = futures[future]
                try:
                    cleaned_batch = future.result()
                    db_manager.insert_issues(cleaned_batch)
                    cleaned_count += len(cleaned_batch)
                    print(
                        f"✅ Inserted {cleaned_count}/{len(raw_issues)} cleaned issues..."
                    )
                except Exception as e:
                    print(f"❌ Error processing batch {batch_index}: {e}")
    finally:
        db_manager.close()

    print(f"🎉 Stored {cleaned_count} issues in the database.")

    # Time tracking
    total_time = time.time() - start_time
    print(f"⏳ Dataset generation completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    generate_dataset()
