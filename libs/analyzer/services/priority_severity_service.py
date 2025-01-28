import torch
from tqdm import tqdm
from transformers import pipeline
from libs.analyzer.db_utils import fetch_all_issues, update_priorities_and_severities
from libs.analyzer.services.fine_tuning import load_priority_severity_model


def predict_priority_and_severity():
    """
    Fetch all issues from the database, predict their priorities and severities using a fine-tuned model first.
    Use zero-shot classification as a fallback for unresolved issues.
    """
    print("Fetching all issues from the database...")
    issues = fetch_all_issues()

    if not issues:
        print("No issues found in the database.")
        return

    print(f"Processing {len(issues)} issues...")

    priority_labels = ["low", "medium", "high"]
    severity_labels = ["minor", "major", "critical"]

    # First Pass: Use the Fine-Tuned Model
    print("Loading the fine-tuned model...")
    unresolved_issues = []
    results = []

    try:
        model, tokenizer = load_priority_severity_model()
        print("Fine-tuned model loaded successfully.")

        for issue in tqdm(issues, desc="Processing issues with fine-tuned model"):
            issue_id, github_id, embedding, title, body, labels, priority, severity = issue
            text = f"{title}\n{body or ''}"

            # Predict using the fine-tuned model
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                logits = model(**inputs).logits
                probabilities = torch.sigmoid(logits).numpy()[0]

            # Extract priority and severity predictions
            priority_probs = probabilities[:len(priority_labels)]
            severity_probs = probabilities[len(priority_labels):]

            # Predict priority
            if max(priority_probs) > 0.5:
                predicted_priority = priority_labels[priority_probs.argmax()]
            else:
                predicted_priority = None

            # Predict severity
            if max(severity_probs) > 0.5:
                predicted_severity = severity_labels[severity_probs.argmax()]
            else:
                predicted_severity = None

            # If unresolved, add to unresolved_issues list
            if predicted_priority is None or predicted_severity is None:
                unresolved_issues.append(issue)
            else:
                results.append((github_id, predicted_priority, predicted_severity))

    except Exception as e:
        print(f"Failed to load fine-tuned model: {e}. Skipping to zero-shot classification for all issues.")
        unresolved_issues = issues  # If fine-tuned model fails, process all issues with zero-shot

    # Second Pass: Use Zero-Shot Classification for Unresolved Issues
    if unresolved_issues:
        print(f"Using zero-shot classification for {len(unresolved_issues)} unresolved issues...")
        zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        for issue in tqdm(unresolved_issues, desc="Processing issues with zero-shot classifier"):
            issue_id, github_id, embedding, title, body, labels, priority, severity = issue
            text = f"{title}\n{body or ''}"

            # Predict priority with zero-shot
            priority_predictions = zero_shot_classifier(text, priority_labels)
            predicted_priority = max(
                zip(priority_predictions["labels"], priority_predictions["scores"]),
                key=lambda x: x[1]
            )[0]

            # Predict severity with zero-shot
            severity_predictions = zero_shot_classifier(text, severity_labels)
            predicted_severity = max(
                zip(severity_predictions["labels"], severity_predictions["scores"]),
                key=lambda x: x[1]
            )[0]

            results.append((github_id, predicted_priority, predicted_severity))

    update_priorities_and_severities(results)
    print("Priority and severity predictions updated successfully.")
