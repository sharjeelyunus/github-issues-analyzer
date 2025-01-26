import torch
import pickle
from tqdm import tqdm
from config import LABELS_MODEL, LABELS_THRESHOLD, MODEL_NAME
from db_utils import fetch_all_issues, store_issue_labels
from services.github_service import fetch_repo_labels
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from utils import compute_cosine_similarity

# Initialize your embedding model
EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)

def create_classifier_mps(model_name: str):
    """Create a zero-shot classification pipeline, using MPS if available."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
    else:
        device = torch.device("cpu")

    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        multi_label=True,
    )
    return classifier

classifier = create_classifier_mps(LABELS_MODEL)

def assign_labels_to_issues(batch_size=16):
    print("Fetching labels from repository...")
    labels = fetch_repo_labels()
    if not labels:
        print("No labels found in the repository.")
        return

    print(f"Fetched {len(labels)} labels.")
    # Create a mapping from label name -> label description
    enriched_labels = {label["name"]: label["description"] for label in labels}
    label_names = list(enriched_labels.keys())
    label_descriptions = list(enriched_labels.values())

    # Create embeddings for each label description (so we can do a similarity check)
    print("Generating embeddings for label descriptions...")
    label_embeddings = {}
    for name, desc in enriched_labels.items():
        text_for_embedding = desc if desc else name
        label_embeddings[name] = EMBEDDING_MODEL.encode(text_for_embedding)

    print("Fetching issues from the database...")
    issues = fetch_all_issues()
    if not issues:
        print("No issues found in the database.")
        return

    # We'll keep track of issues that didn't match any label via embeddings
    unmatched_issues = []
    unmatched_texts = []

    print("Assigning labels via embeddings (first pass)...")
    for (issue_id, github_id, embedding_pickle, title, body) in tqdm(issues, desc="Embedding Check"):
        assigned_label = None
        if embedding_pickle:
            try:
                issue_embedding = pickle.loads(embedding_pickle)

                # Compare similarity to each label and pick the single best match
                sim_scores = []
                for lbl_name, lbl_emb in label_embeddings.items():
                    sim = compute_cosine_similarity(issue_embedding, lbl_emb)
                    sim_scores.append((lbl_name, sim))

                # Sort by descending similarity
                sim_scores.sort(key=lambda x: x[1], reverse=True)

                top_label, top_sim = sim_scores[0]
                if top_sim >= LABELS_THRESHOLD:
                    assigned_label = top_label
                    store_issue_labels(github_id, [assigned_label])

            except Exception as e:
                print(f"Error unpickling embedding for issue {issue_id}: {e}")

        # If we didn't assign a label from embedding, queue it for zero-shot
        if not assigned_label:
            text_content = f"{(title or '')}. {(body or '')}".strip()
            unmatched_issues.append((issue_id, github_id, title, body))
            unmatched_texts.append(text_content)

    # Run zero-shot classification *only* on the unmatched issues
    if unmatched_issues:
        print(f"Running zero-shot classification for {len(unmatched_issues)} unmatched issues...")
        num_batches = (len(unmatched_issues) + batch_size - 1) // batch_size

        # Use tqdm so we can see progress during zero-shot classification too
        for b in tqdm(range(num_batches), desc="Zero-Shot Classification"):
            start_idx = b * batch_size
            end_idx = start_idx + batch_size

            batch_unmatched = unmatched_issues[start_idx:end_idx]
            batch_texts = unmatched_texts[start_idx:end_idx]

            # Zero-shot classification in a single call for this batch
            zsc_results = classifier(batch_texts, label_descriptions)

            # Iterate over each issue in this batch
            for (issue_info, result) in zip(batch_unmatched, zsc_results):
                issue_id, github_id, title, body = issue_info
                zsc_scores = result["scores"]
                label_score_pairs = list(zip(label_names, zsc_scores))
                label_score_pairs.sort(key=lambda x: x[1], reverse=True)

                # Single best label approach
                best_label, best_score = label_score_pairs[0]
                if best_score > LABELS_THRESHOLD:
                    store_issue_labels(github_id, [best_label])

    print("Done assigning labels.")
