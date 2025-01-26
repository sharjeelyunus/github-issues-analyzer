import sqlite3
from transformers import pipeline
from config import LABELS_THRESHOLD
from db_utils import fetch_all_issues, store_issue_labels
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

    enriched_labels = {label["name"]: label["description"] for label in labels}

    print("Fetching issues from the database...")
    issues = fetch_all_issues()
    if not issues:
        print("No issues found in the database.")
        return

    for issue_id, github_id, embedding, title, body in issues:
        if not title and not body:
            print(f"Issue #{github_id} has no title or body. Skipping...")
            continue

        # Perform zero-shot classification
        text = f"{title}. {body}"
        result = classifier(text, list(enriched_labels.values()))

        # Filter labels with a confidence score above the threshold
        assigned_labels = [
            list(enriched_labels.keys())[i]
            for i, score in enumerate(result["scores"])
            if score > LABELS_THRESHOLD
        ]

        if assigned_labels:
            store_issue_labels(github_id, assigned_labels)
