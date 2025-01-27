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

def fine_tune_labels_model(
    issues: List[Tuple[int, int, str, str, str, str]],
    labels: List[Dict[str, str]],
    train_ratio: float = 0.8,
) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    """
    Fine-tune a BERT-based model for multi-label classification on existing labeled issues.
    """
    print("Preparing dataset for fine-tuning...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    label_names = [label["name"] for label in labels]

    # Validate and parse JSON labels
    validated_issues = []
    for _, _, _, title, body, lbls_str, _, _ in issues:
        if not lbls_str or lbls_str == "[]":
            continue
        try:
            lbls_list = json.loads(lbls_str)
            if isinstance(lbls_list, list) and len(lbls_list) > 0:
                validated_issues.append((title, body, lbls_list))
        except json.JSONDecodeError:
            continue

    if not validated_issues:
        print("No valid labeled issues found. Skipping fine-tuning.")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(label_names),
            problem_type="multi_label_classification",
        )
        return model, tokenizer

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

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )

    device = get_device()
    model.to(device)

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

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning model...")
    trainer.train()
    print("Model fine-tuned.")

    return model, tokenizer
