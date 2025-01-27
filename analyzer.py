import json
from services.fine_tuning import fine_tune_model
from services.github_service import fetch_github_issues, fetch_repo_labels
from services.duplicate_service import find_duplicates
from db_utils import initialize_db, populate_dataset, fetch_all_dataset_issues, store_issue
from services.labeling_service import assign_labels_to_issues
from services.priority_severity_service import predict_priority_and_severity
from utils import extract_labels_from_issues, get_embedding

# Initialize
initialize_db()

def sync_issues():
    """Fetch issues from GitHub and the dataset, fine-tune the model, and process issues."""
    # Populate the dataset with issues from top GitHub repositories
    print("Populating the dataset with issues from top repositories...")
    populate_dataset()

    # Fetch issues from the dataset for fine-tuning
    print("Fetching issues from the dataset for initial fine-tuning...")
    dataset_issues = fetch_all_dataset_issues()
    if dataset_issues:
        print(f"Fine-tuning model on {len(dataset_issues)} dataset issues...")
        labels = extract_labels_from_issues(dataset_issues)
        print(f"Extracted {len(labels)} unique labels from dataset issues.")
        fine_tune_model(dataset_issues, labels)
    else:
        print("No issues found in the dataset. Skipping initial fine-tuning.")

    # Fetch issues from the GitHub repository and store them
    print("Fetching open GitHub issues...")
    repo_issues = fetch_github_issues()
    new_count = 0

    for issue in repo_issues:
        text = f"{issue['title']}\n{issue.get('body', '') or ''}"
        embedding = get_embedding(text)

        if store_issue(issue, embedding):
            new_count += 1

    print(f"Imported {new_count} new issues.")

    # Fetch repository labels for second fine-tuning
    print("Fetching repository labels...")
    repo_labels = fetch_repo_labels()
    if repo_labels:
        print(f"Fine-tuning model on repository issues with {len(repo_labels)} labels...")
        fine_tune_model(repo_issues, repo_labels)
    else:
        print("No labels found in the repository. Skipping fine-tuning on repo issues.")

    predict_priority_and_severity()
    find_duplicates()
    assign_labels_to_issues()

    print("Sync completed.")

