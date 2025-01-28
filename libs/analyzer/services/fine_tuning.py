# fine_tuning.py
import json
import os
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
from config import LABELS_MODEL_DIR, PRIORITY_SEVERITY_MODEL_DIR
from utils import get_device
from collections import Counter


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


def validate_and_prepare_issues(
    issues: List[Dict], label_names: List[str]
) -> Tuple[List[str], List[List[float]]]:
    """
    Validate and prepare issues for fine-tuning.

    Args:
        issues: List of issues to validate and prepare.
        label_names: List of all possible labels.

    Returns:
        Tuple of (texts, targets) for training.
    """
    validated_issues = []

    for issue in issues:
        try:
            title = issue.get("title", "")
            body = issue.get("body", "")
            lbls_str = issue.get("labels", "[]")
            lbls_list = json.loads(lbls_str) if isinstance(lbls_str, str) else lbls_str

            if isinstance(lbls_list, list) and lbls_list:
                validated_issues.append((title, body, lbls_list))
        except (KeyError, json.JSONDecodeError):
            continue

    if not validated_issues:
        raise ValueError("No valid labeled issues found.")

    # Prepare texts and targets
    texts = [f"{title}. {body}" for title, body, _ in validated_issues]
    targets = [
        [1.0 if label in lbls_list else 0.0 for label in label_names]
        for _, _, lbls_list in validated_issues
    ]

    return texts, targets


def fine_tune_model(
    issues: List[Dict],
    labels: List[Dict[str, str]],
    model=None,
    train_ratio: float = 0.8,
    save_path: str = LABELS_MODEL_DIR,
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Fine-tune a BERT-based model for multi-label classification on labeled issues.
    """
    if not labels:
        raise ValueError("No labels provided. Fine-tuning aborted.")

    if not issues:
        raise ValueError("No issues provided. Fine-tuning aborted.")

    print(f"Preparing dataset with {len(labels)} labels for fine-tuning...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_names = [label["name"] for label in labels]

    # Validate and prepare issues
    texts, targets = validate_and_prepare_issues(issues, label_names)

    # Tokenize the dataset
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    encodings["labels"] = torch.tensor(targets, dtype=torch.float)

    dataset = Dataset.from_dict(encodings)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Ensure the model has the correct number of labels
    if model is None:
        print("No preloaded model provided. Initializing a new model...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(label_names),
            problem_type="multi_label_classification",
        )
    else:
        # Check if the model's `num_labels` matches the current dataset
        config = model.config
        if config.num_labels != len(label_names):
            print(f"Adjusting model for {len(label_names)} labels...")
            config.num_labels = len(label_names)
            model = BertForSequenceClassification(config)

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


def compute_metrics_priority_severity(eval_pred):
    """
    Compute classification metrics for priority and severity fine-tuning.
    """
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).float()
    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    precision = precision_score(labels, preds, average="micro", zero_division=0)
    recall = recall_score(labels, preds, average="micro", zero_division=0)

    # Class-specific metrics for debugging
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0).tolist()
    per_class_precision = precision_score(
        labels, preds, average=None, zero_division=0
    ).tolist()
    per_class_recall = recall_score(
        labels, preds, average=None, zero_division=0
    ).tolist()

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def fine_tune_priority_severity_model(
    issues: List[Dict],
    train_ratio: float = 0.8,
    save_path: str = "models/priority_severity_model_v1",
    max_length: int = 512,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 5,
    batch_size: int = 16,
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Fine-tune a BERT-based model for priority and severity classification.

    Args:
        issues: List of issues with text and labels (priority/severity).
        train_ratio: Ratio of training data to validation data.
        save_path: Path to save the fine-tuned model.
        max_length: Max token length for BERT input.
        learning_rate: Learning rate for the optimizer.
        num_train_epochs: Number of epochs to train the model.
        batch_size: Batch size for training.

    Returns:
        Fine-tuned model and tokenizer.
    """
    if not issues:
        raise ValueError("No issues provided for fine-tuning.")

    print(
        f"Preparing dataset with {len(issues)} issues for priority and severity fine-tuning..."
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Extract labels dynamically
    priority_labels = list(
        {issue["priority"] for issue in issues if "priority" in issue}
    )
    severity_labels = list(
        {issue["severity"] for issue in issues if "severity" in issue}
    )

    print(f"Detected priority labels: {priority_labels}")
    print(f"Detected severity labels: {severity_labels}")

    # Preprocess issues into text and binary labels
    texts = [f"{issue['title']} {issue['body']}" for issue in issues]
    targets = [
        [1.0 if priority == issue["priority"] else 0.0 for priority in priority_labels]
        + [
            1.0 if severity == issue["severity"] else 0.0
            for severity in severity_labels
        ]
        for issue in issues
    ]

    # Tokenize the inputs
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    encodings["labels"] = torch.tensor(targets, dtype=torch.float)

    # Split into training and validation datasets
    dataset = Dataset.from_dict(encodings)
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Class weights for priority and severity
    priority_counts = Counter([issue["priority"] for issue in issues])
    severity_counts = Counter([issue["severity"] for issue in issues])

    priority_weights = [1.0 / priority_counts[label] for label in priority_labels]
    severity_weights = [1.0 / severity_counts[label] for label in severity_labels]

    class_weights = torch.tensor(priority_weights + severity_weights, dtype=torch.float)

    # Initialize the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(priority_labels) + len(severity_labels),
        problem_type="multi_label_classification",
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
        compute_metrics=compute_metrics_priority_severity,
        tokenizer=tokenizer,
    )

    print("Fine-tuning priority and severity model...")
    trainer.train()
    print("Model fine-tuned successfully.")

    # Save the model and tokenizer
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}.")

    return model, tokenizer


def load_fine_tuned_model(model_path=LABELS_MODEL_DIR):
    """
    Load a previously fine-tuned model and tokenizer from the specified directory.
    Raises an exception if the model is not found.
    """
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. Please ensure the model is fine-tuned and saved."
        )

    try:
        print(f"Loading model and tokenizer from '{model_path}'...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        print(f"Model and tokenizer successfully loaded from '{model_path}'.")
        return model, tokenizer
    except Exception as e:
        raise Exception(f"Failed to load model from '{model_path}': {e}")


# Load the fine-tuned priority and severity model
def load_priority_severity_model(model_path=PRIORITY_SEVERITY_MODEL_DIR):
    """
    Load the fine-tuned priority and severity prediction model.
    """
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory '{model_path}' not found. Please ensure the model is fine-tuned and saved."
        )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer
