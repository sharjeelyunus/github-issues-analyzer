from transformers import pipeline
from libs.analyzer.db_utils import fetch_all_issues, update_priorities_and_severities
from tqdm import tqdm

def predict_priority_and_severity():
    """
    Fetch all issues from the database, predict their priorities and severities using zero-shot classification,
    and update the database.
    """
    print("Fetching all issues from the database...")
    issues = fetch_all_issues()

    if not issues:
        print("No issues found in the database.")
        return

    print(f"Processing {len(issues)} issues...")

    # Initialize the zero-shot classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    results = []

    priority_labels = ["low", "medium", "high"]
    severity_labels = ["minor", "major", "critical"]

    for issue in tqdm(issues, desc="Predicting priorities and severities"):
        issue_id, github_id, embedding, title, body, labels, priority, severity = issue
        text = f"{title}\n{body or ''}"

        # Predict priority
        priority_predictions = classifier(text, priority_labels)
        predicted_priority = max(
            zip(priority_predictions["labels"], priority_predictions["scores"]),
            key=lambda x: x[1]
        )[0]

        # Predict severity
        severity_predictions = classifier(text, severity_labels)
        predicted_severity = max(
            zip(severity_predictions["labels"], severity_predictions["scores"]),
            key=lambda x: x[1]
        )[0]

        results.append((github_id, predicted_priority, predicted_severity))

    update_priorities_and_severities(results)
    print("Priority and severity predictions updated successfully.")