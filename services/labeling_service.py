from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
)

from config import LABELS_MODEL, LABELS_THRESHOLD
from db_utils import fetch_all_issues, store_issue_labels
from services.fine_tuning import fine_tune_labels_model
from services.github_service import fetch_repo_labels
from utils import get_device


def create_classifier_mps(model_name: str):
    """
    Create a zero-shot classification pipeline, using GPU/MPS if available, else CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = get_device()
    model.to(device)

    return pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        multi_label=True,
    )


def classify_with_fine_tuned_model(
    issues: List[Tuple[int, int, str, str, str, str]],
    label_names: List[str],
    fine_tuned_model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    batch_size: int = 16
) -> List[Tuple[int, int, str, str, str, str]]:
    """
    Classify issues using the fine-tuned model in batches. Returns any issues that
    did not meet the threshold for at least one label.

    Args:
        issues: List of unlabeled issues (issue_id, github_id, ..., title, body, label_json).
        label_names: List of possible label names for classification.
        fine_tuned_model: The trained multi-label classification model.
        tokenizer: Tokenizer for the model.
        batch_size: Number of items to process in one batch.

    Returns:
        List of issues that did not get any label assigned.
    """
    device = get_device()
    fine_tuned_model.to(device)
    fine_tuned_model.eval()

    issues_without_labels = []

    for start_idx in tqdm(range(0, len(issues), batch_size), desc="Fine-Tune Classification"):
        batch = issues[start_idx:start_idx + batch_size]

        # Prepare texts for the batch
        batch_texts = [f"{(title or '')}. {(body or '')}" for _, _, _, title, body, _ in batch]
        encodings = tokenizer(
            batch_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = fine_tuned_model(**encodings)
            predictions = outputs.logits.sigmoid().cpu().numpy() 

        # Determine assigned labels per issue
        for (issue, pred_scores) in zip(batch, predictions):
            issue_id, github_id, _, title, body, _ = issue
            assigned_labels = [
                label_names[i] for i, score in enumerate(pred_scores) if score > LABELS_THRESHOLD
            ]
            if assigned_labels:
                store_issue_labels(github_id, assigned_labels)
            else:
                issues_without_labels.append(issue)

    return issues_without_labels


def classify_with_zero_shot(
    issues_without_labels: List[Tuple[int, int, str, str, str, str]],
    enriched_labels: List[str],
    label_names: List[str],
    batch_size: int = 16
) -> None:
    """
    Classify issues without labels using zero-shot classification in batches.
    Only the single best label is assigned if above LABELS_THRESHOLD.
    """
    classifier = create_classifier_mps(LABELS_MODEL)
    print(f"Classifying {len(issues_without_labels)} issues with zero-shot classification...")

    for start_idx in tqdm(range(0, len(issues_without_labels), batch_size), desc="Zero-Shot Classification"):
        batch = issues_without_labels[start_idx:start_idx + batch_size]
        batch_texts = [f"{(title or '')}. {(body or '')}".strip() for _, _, _, title, body, _ in batch]

        # Perform classification
        zsc_results = classifier(batch_texts, enriched_labels)
        for issue, result in zip(batch, zsc_results):
            issue_id, github_id, _, title, body, _ = issue
            zsc_scores = result["scores"]
            label_score_pairs = list(zip(label_names, zsc_scores))
            label_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Pick the single best label
            best_label, best_score = label_score_pairs[0]
            if best_score > LABELS_THRESHOLD:
                store_issue_labels(github_id, [best_label])


def assign_labels_to_issues() -> None:
    """
    Main entry point for fetching issues, training a fine-tuned model,
    classifying new issues, and fallback to zero-shot classification if needed.
    """
    print("Fetching labels from repository...")
    labels = fetch_repo_labels()
    if not labels:
        print("No labels found in the repository.")
        return

    print(f"Fetched {len(labels)} labels.")
    enriched_labels = [f"{label['name']}: {label['description']}" for label in labels]
    label_names = [label["name"] for label in labels]

    print("Fetching issues from the database...")
    issues = fetch_all_issues()
    if not issues:
        print("No issues found in the database.")
        return

    # Fine-tune model with existing labeled issues
    fine_tuned_model, tokenizer = fine_tune_labels_model(issues, labels)

    # Filter issues to process only those without existing labels
    issues_to_process = [issue for issue in issues if not issue[-1] or issue[-1] == "[]"]
    if not issues_to_process:
        print("No issues without labels to process.")
        return

    print(f"Classifying {len(issues_to_process)} issues...")

    # Classify issues using the fine-tuned model
    issues_without_labels = classify_with_fine_tuned_model(
        issues_to_process, label_names, fine_tuned_model, tokenizer
    )

    # Classify remaining issues with zero-shot classification (best single label only)
    if issues_without_labels:
        classify_with_zero_shot(issues_without_labels, enriched_labels, label_names)

    print("Label assignment completed.")
