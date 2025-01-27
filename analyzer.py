from services.github_service import fetch_github_issues
from services.duplicate_service import find_duplicates
from db_utils import initialize_db, store_issue
from services.labeling_service import assign_labels_to_issues
from services.priority_severity_service import predict_priority_and_severity
from utils import get_embedding

# Initialize
initialize_db()

def sync_issues():
    """Fetch issues from GitHub, store them in the database, detect duplicates, and assign labels."""
    print("Fetching open GitHub issues...")
    issues = fetch_github_issues()
    new_count = 0

    for issue in issues:
        text = f"{issue['title']}\n{issue.get('body', '') or ''}"
        embedding = get_embedding(text)

        if store_issue(issue, embedding):
            new_count += 1

    print(f"Imported {new_count} new issues.")
    predict_priority_and_severity()
    find_duplicates()
    assign_labels_to_issues()
    print("Done.")
