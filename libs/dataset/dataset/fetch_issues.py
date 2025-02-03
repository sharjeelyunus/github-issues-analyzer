import os
import time
import json
import pyarrow as pa
import pyarrow.parquet as pq
from config import DATASET_REPOS_COUNT, GITHUB_TOKENS, RAW_ISSUES_FILE, TOP_REPOS_FILE
from libs.dataset.dataset.github_fetcher import GitHubFetcher


def write_parquet_in_batches(data, output_file, batch_size=100000):
    """
    Writes a list of dictionaries to a Parquet file in batches,
    enforcing a consistent schema across all batches.

    Args:
        data (list): List of dictionaries (raw issues).
        output_file (str): The Parquet file path.
        batch_size (int): Number of records to process at a time.
    """
    writer = None
    total_records = len(data)
    schema = None

    # Normalize a record against a given schema.
    def normalize_record(record, schema):
        normalized = {}
        for field in schema.names:
            # If the field is missing in the record, set it to None.
            normalized[field] = record.get(field, None)
        return normalized

    for i in range(0, total_records, batch_size):
        batch = data[i : i + batch_size]

        if schema is None:
            # For the first batch, infer the schema.
            table = pa.Table.from_pylist(batch)
            schema = table.schema
            writer = pq.ParquetWriter(output_file, schema, compression="snappy")
        else:
            # Normalize each record in the batch so that it contains all schema fields.
            batch = [normalize_record(record, schema) for record in batch]
            try:
                table = pa.Table.from_pylist(batch, schema=schema)
            except Exception as e:
                print(f"❌ Error converting batch {i} with enforced schema: {e}")
                raise e  # Reraise or handle as needed.

        writer.write_table(table)
        print(f"Wrote records {i} to {min(i+batch_size, total_records)}")

    if writer:
        writer.close()


def fetch_and_store_issues():
    start_time = time.time()

    fetcher = GitHubFetcher(GITHUB_TOKENS, max_workers=20, per_repo_workers=8)

    # Check if the top repositories file exists.
    if os.path.exists(TOP_REPOS_FILE):
        print(f"📂 Loading top repositories from `{TOP_REPOS_FILE}`...")
        with open(TOP_REPOS_FILE, "r", encoding="utf-8") as f:
            top_repos = json.load(f)
        print(f"✅ Loaded {len(top_repos)} top repositories from file.")
    else:
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

    # Write the data to a Parquet file in batches to avoid memory issues
    try:
        write_parquet_in_batches(raw_issues, RAW_ISSUES_FILE, batch_size=100000)
        print(f"📂 Stored raw issues in Parquet format at `{RAW_ISSUES_FILE}`.")
    except Exception as e:
        print(f"❌ Error writing Parquet file: {e}")

    total_time = time.time() - start_time
    print(f"⏳ Fetching completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    fetch_and_store_issues()
