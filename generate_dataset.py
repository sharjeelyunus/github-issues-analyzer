import os
import time
from config import RAW_ISSUES_FILE
from libs.dataset.dataset.fetch_issues import fetch_and_store_issues
from libs.dataset.dataset.process_issues import clean_and_store_issues

def generate_dataset():
    start_time = time.time()

    # Step 1: Fetch and store raw issues only if the file doesn't exist.
    if os.path.exists(RAW_ISSUES_FILE):
        print(f"📂 Raw issues file '{RAW_ISSUES_FILE}' already exists. Skipping fetch.")
    else:
        print("🔍 Raw issues file not found. Fetching raw issues...")
        fetch_and_store_issues()

    # Step 2: Clean and store issues in the database.
    clean_and_store_issues()

    # Time tracking
    total_time = time.time() - start_time
    print(f"⏳ Dataset generation completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    generate_dataset()
