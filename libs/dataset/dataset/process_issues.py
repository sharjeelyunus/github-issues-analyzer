import time
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import DB_FILE, RAW_ISSUES_FILE
from libs.dataset.dataset.issue_cleaner import clean_issues
from libs.dataset.dataset.database_manager import DatabaseManager

def clean_and_store_issues():
    start_time = time.time()

    # Load raw issues from the Parquet file
    print(f"📂 Loading raw issues from `{RAW_ISSUES_FILE}`...")
    try:
        # Read the Parquet file
        table = pq.read_table(RAW_ISSUES_FILE)
        raw_issues = table.to_pylist()
    except Exception as e:
        print(f"❌ Error loading issues: {e}")
        return

    print(f"✅ Loaded {len(raw_issues)} raw issues for processing.")

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
                    print(f"✅ Inserted {cleaned_count}/{len(raw_issues)} cleaned issues...")
                except Exception as e:
                    print(f"❌ Error processing batch {batch_index}: {e}")
    finally:
        db_manager.close()

    print(f"🎉 Stored {cleaned_count} issues in the database.")

    # Time tracking
    total_time = time.time() - start_time
    print(f"⏳ Cleaning & storing completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    clean_and_store_issues()
