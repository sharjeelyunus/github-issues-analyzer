import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
from fuzzywuzzy import process

def get_fuzzy_metric(labels, mapping, threshold=80):
    """
    Determine priority based on fuzzy matching against a flexible list of possible labels.

    Args:
        labels (list): A list of labels from the issue.
        mapping (dict): A dictionary with priority levels as keys and possible labels as values.
        threshold (int): The similarity threshold (0-100) for fuzzy matching.

    Returns:
        str or None: The determined priority level or None if no match is found.
    """
    for label in labels:
        normalized_label = label.lower()
        for priority, keywords in mapping.items():
            match, score = process.extractOne(normalized_label, keywords)
            if score >= threshold:
                return priority
    return None


def get_issue_metric(labels, mapping):
    """
    Determine an issue metric (e.g., priority or severity) based on a partial match in its labels.

    Args:
        labels (list): A list of labels associated with the issue.
        mapping (dict): A dictionary mapping label keywords (case-insensitive) to metric values.

    Returns:
        str or None: The determined metric value, or None if no matching keyword is found.
    """
    for label in labels:
        normalized_label = label.lower()
        for key in mapping:
            if key in normalized_label:
                return mapping[key]
    return None


def compute_engagement_metric(issue, mapping, comment_weight=2, reaction_weight=1):
    """
    Compute an engagement-based metric (e.g., priority or severity) for an issue.

    Args:
        issue (dict): The issue data containing comments and reactions.
        mapping (dict): A mapping dictionary to translate engagement thresholds to labels.
        comment_weight (int, optional): Weight assigned to comments. Default is 2.
        reaction_weight (int, optional): Weight assigned to reactions. Default is 1.

    Returns:
        str or None: The computed metric value based on the mapping (e.g., "high", "medium", "low").
    """
    comments = issue.get("comments", 0)
    reactions = issue.get("reactions", {}).get("total_count", 0)

    # Compute the engagement score with weights
    engagement_score = (comments * comment_weight) + (reactions * reaction_weight)

    # Determine the label using the provided mapping
    for threshold, label in sorted(mapping.items(), reverse=True):
        if engagement_score >= threshold:
            return label

    return None


def predict_with_model(issue, mapping, max_length=512):
    """
    Predict a metric (e.g., priority or severity) using a lightweight model (e.g., DistilBERT) with truncation.

    Args:
        issue (dict): The issue data containing 'title' and 'body'.
        mapping (dict): A dictionary mapping model output classes to labels.
        model: The trained model for prediction.
        tokenizer: The tokenizer corresponding to the model.
        max_length (int, optional): Maximum length for tokenization. Default is 512.

    Returns:
        str or None: The predicted label based on the mapping, or None if text is empty.
    """
    # Combine title and body for prediction
    text = f"{issue.get('title', '')} {issue.get('body', '')}".strip()
    if not text:
        return None

    # Tokenize the text with truncation
    inputs = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Use the model to make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).squeeze()

    # Map model outputs to labels
    predicted_class = probabilities.argmax().item()
    return mapping.get(predicted_class, None)
