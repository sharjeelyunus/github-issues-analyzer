# libs/analyzer/analyzer.py

from libs.analyzer.services.fine_tuning import fine_tune_model, load_fine_tuned_model
from libs.analyzer.services.github_service import fetch_github_issues, fetch_repo_labels
from libs.analyzer.services.duplicate_service import find_duplicates
from libs.analyzer.db_utils import initialize_db, store_issue
from libs.analyzer.services.labeling_service import assign_labels_to_issues
from libs.analyzer.services.priority_severity_service import predict_priority_and_severity
from utils import get_embedding

def sync_issues():
    """Fetch issues from GitHub, fine-tune the model on repo issues, and process them."""
    initialize_db()
    
    print("Loading pre-fine-tuned model for label prediction...")
    model, _ = load_fine_tuned_model()
    if not model:
        raise Exception("Label prediction model not found. Run fine_tune_dataset.py first.")

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

    # Fetch repository labels for fine-tuning
    print("Fetching repository labels...")
    repo_labels = fetch_repo_labels()
    if repo_labels:
        print(f"Fine-tuning label prediction model on repository issues with {len(repo_labels)} labels...")
        fine_tune_model(repo_issues, repo_labels, model)
    else:
        print("No labels found in the repository. Skipping fine-tuning on repo issues.")

    # Perform analysis
    predict_priority_and_severity()
    find_duplicates()
    assign_labels_to_issues()

    print("Sync completed.")
