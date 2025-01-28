from libs.analyzer.services.github_service import fetch_github_issues
from libs.analyzer.services.duplicate_service import find_duplicates
from libs.analyzer.db_utils import initialize_db, store_issue
from libs.analyzer.services.labeling_service import assign_labels_to_issues
from libs.analyzer.services.priority_severity_service import predict_priority_and_severity
from utils import get_embedding

def sync_issues():
    """Fetch issues from GitHub, and process them with the fine-tuned models."""
    initialize_db()

    # Fetch issues from the GitHub repository
    print("Fetching open GitHub issues...")
    repo_issues = fetch_github_issues()
    new_count = 0

    for issue in repo_issues:
        text = f"{issue['title']}\n{issue.get('body', '') or ''}"
        embedding = get_embedding(text)

        if store_issue(issue, embedding):
            new_count += 1

    print(f"Imported {new_count} new issues.")

    # Perform analysis
    predict_priority_and_severity()
    find_duplicates()
    assign_labels_to_issues()

    print("Sync completed.")
