import os
import time
import json
import pyarrow as pa
import pyarrow.parquet as pq
from config import DATASET_REPOS_COUNT, GITHUB_TOKENS, RAW_ISSUES_FILE, TOP_REPOS_FILE
from libs.dataset.dataset.github_fetcher import GitHubFetcher

def fetch_and_store_issues():
    start_time = time.time()

    fetcher = GitHubFetcher(GITHUB_TOKENS, max_workers=20, per_repo_workers=8)

    # Check if the top repositories file exists.
    if os.path.exists(TOP_REPOS_FILE):
        # Load top repositories from the file.
        print(f"📂 Loading top repositories from `{TOP_REPOS_FILE}`...")
        with open(TOP_REPOS_FILE, "r", encoding="utf-8") as f:
            top_repos = json.load(f)
        print(f"✅ Loaded {len(top_repos)} top repositories from file.")
    else:
        # Fetch the top repositories and save them.
        print("Fetching top repositories...")
        top_repos = fetcher.fetch_top_repositories(count=DATASET_REPOS_COUNT)
        print(f"✅ Fetched {len(top_repos)} repositories.")
        with open(TOP_REPOS_FILE, "w", encoding="utf-8") as f:
            json.dump(top_repos, f, indent=2)
        print(f"📂 Stored top repositories in `{TOP_REPOS_FILE}`.")

    # Fetch issues concurrently using the top repositories.
    print(f"Fetching issues for {len(top_repos)} repositories concurrently...")
    raw_issues = fetcher.fetch_issues_concurrently(top_repos)
    print(f"✅ Fetched {len(raw_issues)} raw issues.")

    # If raw_issues is a nested list, flatten it.
    if raw_issues and isinstance(raw_issues[0], list):
        raw_issues = [issue for repo_issues in raw_issues for issue in repo_issues]

    # Convert the list of dictionaries to a PyArrow Table.
    try:
        table = pa.Table.from_pylist(raw_issues)
    except Exception as e:
        print(f"❌ Error converting raw issues to a table: {e}")
        return

    # Write the table to a Parquet file with Snappy compression.
    try:
        pq.write_table(table, RAW_ISSUES_FILE, compression="snappy")
        print(f"📂 Stored raw issues in Parquet format at `{RAW_ISSUES_FILE}`.")
    except Exception as e:
        print(f"❌ Error writing Parquet file: {e}")

    # Time tracking.
    total_time = time.time() - start_time
    print(f"⏳ Fetching completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    fetch_and_store_issues()
