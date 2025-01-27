import json
from typing import List, Tuple, Dict, Optional

import torch
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BertTokenizer,
    BertForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from config import LABELS_MODEL, LABELS_THRESHOLD
from db_utils import fetch_all_issues, store_issue_labels
from services.github_service import fetch_repo_labels


def get_device() -> torch.device:
    """
    Return the best available device among CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_classifier_mps(model_name: str):
    """
    Create a zero-shot classification pipeline, using GPU/MPS if available, else CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = get_device()
    model.to(device)

    # We keep device_map='auto' for MPS compatibility:
    return pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # handle MPS automatically
        multi_label=True,
    )


def compute_metrics(eval_pred):
    """
    Compute multi-label classification metrics: precision, recall, F1.
    Uses a 0.5 threshold on sigmoid outputs.
    """
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Sigmoid and threshold at 0.5
    preds = (torch.sigmoid(logits) > 0.5).float()

    preds_np = preds.numpy()
    labels_np = labels.numpy()

    # 'micro' average is common for multi-label tasks
    f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
    precision = precision_score(labels_np, preds_np, average="micro", zero_division=0)
    recall = recall_score(labels_np, preds_np, average="micro", zero_division=0)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def fine_tune_model(
    issues: List[Tuple[int, int, str, str, str, str]],
    labels: List[Dict[str, str]],
    train_ratio: float = 0.8
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Fine-tune a BERT-based model for multi-label classification on existing labeled issues.

    Args:
        issues: List of issues. Each issue is a tuple like (issue_id, github_id, ?, title, body, label_json).
        labels: List of repository labels (dicts with 'name' and 'description').
        train_ratio: Ratio of data to be used for training (remainder is for validation).

    Returns:
        A tuple of (fine_tuned_model, tokenizer).
    """
    print("Preparing dataset for fine-tuning...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_names = [label["name"] for label in labels]

    # Filter issues that have actual labels
    labeled_issues = [(title, body, lbls) for _, _, _, title, body, lbls in issues
                      if lbls and lbls != "[]"]
    if not labeled_issues:
        print("No labeled issues found. Skipping fine-tuning.")
        # Return a fresh model/tokenizer (or you could handle it differently)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(label_names),
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )
        return model, tokenizer

    texts = [f"{title}. {body}" for (title, body, lbls) in labeled_issues]
    targets = [
        [1.0 if label in json.loads(lbls) else 0.0 for label in label_names]
        for (_, _, lbls) in labeled_issues
    ]

    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    encodings["labels"] = torch.tensor(targets, dtype=torch.float)

    # Create Dataset
    dataset = Dataset.from_dict(encodings)

    # Split into train / validation
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True  # Suppress initialization warnings
    )

    # Move model to best device
    device = get_device()
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Track multi-label metrics
    )

    print("Fine-tuning model...")
    trainer.train()
    print("Model fine-tuned.")

    return model, tokenizer


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

        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = fine_tuned_model(**encodings)
            predictions = outputs.logits.sigmoid().cpu().numpy()  # shape: (batch_size, num_labels)

        # Determine assigned labels per issue
        for (issue, pred_scores) in zip(batch, predictions):
            issue_id, github_id, _, title, body, _ = issue
            assigned_labels = [
                label_names[i] for i, score in enumerate(pred_scores) if score > LABELS_THRESHOLD
            ]
            if assigned_labels:
                store_issue_labels(github_id, assigned_labels)
                print(f"Issue #{github_id} assigned labels by fine-tuned model: {', '.join(assigned_labels)}")
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

        # Perform classification for the whole batch
        zsc_results = classifier(batch_texts, enriched_labels)

        # Each zsc_results[i] is a dict with 'labels' and 'scores'
        for issue, result in zip(batch, zsc_results):
            issue_id, github_id, _, title, body, _ = issue
            zsc_scores = result["scores"]
            label_score_pairs = list(zip(label_names, zsc_scores))
            label_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Pick the single best label if above threshold
            best_label, best_score = label_score_pairs[0]
            if best_score > LABELS_THRESHOLD:
                store_issue_labels(github_id, [best_label])
                print(f"Issue #{github_id} assigned label by zero-shot: {best_label}")


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
    fine_tuned_model, tokenizer = fine_tune_model(issues, labels)

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
