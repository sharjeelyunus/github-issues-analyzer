import sqlite3
from transformers import pipeline
from config import LABELS_THRESHOLD
from db_utils import fetch_all_issues_with_embeddings, store_issue_labels
from services.github_service import fetch_repo_labels

# Initialize the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def assign_labels_to_issues():
    """Assign labels to issues based on their context."""
    print("Fetching labels from repository...")
    labels = fetch_repo_labels()

    if not labels:
        print("No labels found in the repository.")
        return

    print(f"Fetched {len(labels)} labels.")
    print("Fetching issues from the database...")

    issues = fetch_all_issues_with_embeddings()
    if not issues:
        print("No issues found in the database.")
        return

    for issue_id, github_id, embedding in issues:
        # Fetch title and body for the current issue
        conn = sqlite3.connect("issues.db")
        cursor = conn.cursor()
        cursor.execute("SELECT title, body FROM issues WHERE id = ?", (issue_id,))
        issue_data = cursor.fetchone()
        conn.close()

        if issue_data:
            title, body = issue_data
        else:
            print(f"Issue #{github_id} not found in database. Skipping...")
            continue

        # Perform zero-shot classification
        text = f"{title}. {body}"
        result = classifier(text, labels)

        # Filter labels with a confidence score above the threshold
        assigned_labels = [label for label, score in zip(result["labels"], result["scores"]) if score > LABELS_THRESHOLD]

        if assigned_labels:
            store_issue_labels(github_id, assigned_labels)
