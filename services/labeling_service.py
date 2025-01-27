import torch
from tqdm import tqdm
from config import LABELS_MODEL, LABELS_THRESHOLD
from db_utils import fetch_all_issues, store_issue_labels
from services.github_service import fetch_repo_labels
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def create_classifier_mps(model_name: str):
    """Create a zero-shot classification pipeline, using MPS if available."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Use MPS if available, otherwise fallback to CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    return pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        multi_label=True,
    )

classifier = create_classifier_mps(LABELS_MODEL)

def classify_issues(batch, enriched_labels, label_names):
    """Classify a batch of issues using zero-shot classification."""
    batch_texts = [f"{(title or '')}. {(body or '')}".strip() for _, _, _, title, body, _ in batch]
    zsc_results = classifier(batch_texts, enriched_labels)

    for (issue_id, github_id, _, title, body, _), result in zip(batch, zsc_results):
        zsc_scores = result["scores"]
        label_score_pairs = list(zip(label_names, zsc_scores))
        label_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Choose the best label if it meets the threshold
        best_label, best_score = label_score_pairs[0]
        if best_score > LABELS_THRESHOLD:
            store_issue_labels(github_id, [best_label])

def assign_labels_to_issues(batch_size=16):
    """Fetch issues and assign labels using zero-shot classification."""
    print("Fetching labels from repository...")
    labels = fetch_repo_labels()
    if not labels:
        print("No labels found in the repository.")
        return

    print(f"Fetched {len(labels)} labels.")

    # Prepare enriched labels for classification
    enriched_labels = [f"{label['name']}: {label['description']}" for label in labels]
    label_names = [label["name"] for label in labels]

    print("Fetching issues from the database...")
    issues = fetch_all_issues()
    if not issues:
        print("No issues found in the database.")
        return

    # Filter issues to process only those without existing labels
    issues_to_process = [issue for issue in issues if not issue[-1] or issue[-1] == "[]"]

    if not issues_to_process:
        print("No issues without labels to process.")
        return

    print(f"Classifying {len(issues_to_process)} issues...")

    # Process issues in batches
    for batch_start in tqdm(range(0, len(issues_to_process), batch_size), desc="Zero-Shot Classification"):
        batch = issues_to_process[batch_start:batch_start + batch_size]
        classify_issues(batch, enriched_labels, label_names)

    print("Label assignment completed.")
