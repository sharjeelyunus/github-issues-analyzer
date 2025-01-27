import json
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize model
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str):
    """Generate embedding for a given text."""
    return MODEL.encode(text).tolist()


def compute_cosine_similarity(vec_a, vec_b):
    """Compute the cosine similarity between two vectors."""
    return float(util.cos_sim(vec_a, vec_b)[0][0])


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


def extract_labels_from_issues(issues):
    """Extract unique labels from a list of issues."""
    extracted_labels = set()

    for issue in issues:
        try:
            # Handle both tuple and dictionary formats for issues
            lbls_str = (
                issue[5]
                if isinstance(issue, tuple) and len(issue) > 5
                else issue.get("labels", "[]")
            )
            lbls_list = json.loads(lbls_str) if isinstance(lbls_str, str) else lbls_str

            if isinstance(lbls_list, list):
                for lbl in lbls_list:
                    # Extract only the name field
                    extracted_labels.add(lbl["name"] if isinstance(lbl, dict) else lbl)
        except (IndexError, KeyError, json.JSONDecodeError):
            continue

    # Convert the set of labels into a list of dictionaries
    return [
        {"name": label, "description": f"This issue is related to {label}"}
        for label in extracted_labels
    ]
