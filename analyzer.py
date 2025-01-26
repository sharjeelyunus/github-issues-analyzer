from services.github_service import fetch_github_issues
from services.duplicate_service import find_duplicates
from db_utils import initialize_db, store_issue
from config import MODEL_NAME
from sentence_transformers import SentenceTransformer

# Initialize
model = SentenceTransformer(MODEL_NAME)
initialize_db()

def get_embedding(text: str):
    """Generate embedding for a given text."""
    return model.encode(text).tolist()

def sync_issues():
    """Fetch issues from GitHub, store them in the database, and detect duplicates."""
    print("Fetching open GitHub issues...")
    issues = fetch_github_issues()
    new_count = 0

    for issue in issues:
        text = f"{issue['title']}\n{issue.get('body', '') or ''}"
        embedding = get_embedding(text)

        if store_issue(issue, embedding):
            new_count += 1

    print(f"Imported {new_count} new issues.")
    print("Finding duplicates...")
    find_duplicates()
    print("Done.")
