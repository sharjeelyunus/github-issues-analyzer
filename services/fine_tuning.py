# fine_tuning.py
import json
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
from torch.utils.data import random_split
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import List, Tuple, Dict
from db_utils import fetch_all_issues
from services.github_service import fetch_repo_labels
from utils import get_device


def compute_metrics(eval_pred):
    """
    Compute multi-label classification metrics: precision, recall, F1.
    Uses a 0.5 threshold on sigmoid outputs.
    """
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    preds = (torch.sigmoid(logits) > 0.5).float()
    preds_np = preds.numpy()
    labels_np = labels.numpy()

    f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
    precision = precision_score(labels_np, preds_np, average="micro", zero_division=0)
    recall = recall_score(labels_np, preds_np, average="micro", zero_division=0)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


from transformers import BertForSequenceClassification, BertTokenizer

def fine_tune_model(
    issues: List[Tuple],
    labels: List[Dict[str, str]],
    train_ratio: float = 0.8,
    save_path: str = "bert-base-uncased",
    ignore_mismatched_sizes: bool = True
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Fine-tune a BERT-based model for multi-label classification on provided labeled issues.

    Args:
        issues: List of issues to fine-tune on.
        labels: List of all possible labels for classification.
        train_ratio: Ratio of training to validation data.
        save_path: Path to save the fine-tuned model.
        ignore_mismatched_sizes: Whether to ignore size mismatches when loading the model.
    """
    if not labels:
        print("No labels provided. Skipping fine-tuning.")
        return None, None

    if not issues:
        print("No issues provided. Skipping fine-tuning.")
        return None, None

    print(f"Preparing dataset with {len(labels)} labels for fine-tuning...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_names = [label["name"] for label in labels]

    validated_issues = []
    for issue in issues:
        try:
            title = issue[3] if isinstance(issue, tuple) and len(issue) > 3 else issue.get("title", "")
            body = issue[4] if isinstance(issue, tuple) and len(issue) > 4 else issue.get("body", "")
            lbls_str = issue[5] if isinstance(issue, tuple) and len(issue) > 5 else issue.get("labels", "[]")
            lbls_list = json.loads(lbls_str) if isinstance(lbls_str, str) else lbls_str

            if isinstance(lbls_list, list) and len(lbls_list) > 0:
                validated_issues.append((title, body, lbls_list))
        except (IndexError, KeyError, json.JSONDecodeError):
            continue

    if not validated_issues:
        print("No valid labeled issues found. Skipping fine-tuning.")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(label_names),
            problem_type="multi_label_classification",
        )
        return model, tokenizer

    # Prepare the dataset
    texts = [f"{title}. {body}" for title, body, _ in validated_issues]
    targets = [
        [1.0 if label in lbls_list else 0.0 for label in label_names]
        for _, _, lbls_list in validated_issues
    ]

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    encodings["labels"] = torch.tensor(targets, dtype=torch.float)

    dataset = Dataset.from_dict(encodings)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Initialize the model with the correct number of labels
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=ignore_mismatched_sizes,  # Handle size mismatches
    )

    device = get_device()
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    print("Fine-tuning the model...")
    trainer.train()
    print("Model fine-tuned.")

    # Save the model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}.")

    return model, tokenizer


def load_fine_tuned_model(model_path="bert-base-uncased"):
    """
    Load a previously fine-tuned model and tokenizer from disk.
    """
    print(f"Loading fine-tuned model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer
