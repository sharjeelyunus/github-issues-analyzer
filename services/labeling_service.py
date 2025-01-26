import pickle
from config import LABELS_THRESHOLD
from utils import get_embedding, compute_cosine_similarity
from db_utils import fetch_all_issues_with_embeddings, store_issue_labels
from services.github_service import fetch_repo_labels

def assign_labels_to_issues():
    """Assign labels to issues based on their title, body, and comments."""
    print("Fetching labels from repository...")
    labels = fetch_repo_labels()

    if not labels:
        print("No labels found in the repository.")
        return

    enriched_labels = {label: f"This issue is related to {label}" for label in labels}
    label_embeddings = {label: get_embedding(enriched_label.lower()) for label, enriched_label in enriched_labels.items()}

    print(f"Fetched and enriched {len(labels)} labels.")
    print("Fetching issues from the database...")

    issues = fetch_all_issues_with_embeddings()
    if not issues:
        print("No issues found in the database.")
        return

    for issue_id, github_id, embedding in issues:
        issue_embedding = pickle.loads(embedding)
        assigned_labels = []

        print(f"Analyzing Issue #{github_id}...")

        for label, label_embedding in label_embeddings.items():
            similarity = compute_cosine_similarity(issue_embedding, label_embedding)

            print(f"Label: {label}, Similarity: {similarity:.4f}")

            if similarity >= LABELS_THRESHOLD:
                assigned_labels.append(label)

        if assigned_labels:
            print(f"Issue #{github_id} assigned labels: {', '.join(assigned_labels)}")
            store_issue_labels(github_id, assigned_labels)
        else:
            print(f"Issue #{github_id} did not match any labels.")
    print("Label assignment completed.")
