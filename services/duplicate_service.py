import pickle
from db_utils import fetch_all_issues_with_embeddings, update_duplicates
from config import DUPLICATE_THRESHOLD
from utils import compute_cosine_similarity

def find_duplicates():
    """Identify duplicates by analyzing embeddings."""
    issues = fetch_all_issues_with_embeddings()

    embeddings = {row[1]: pickle.loads(row[2]) for row in issues}
    duplicates = {}

    for issue_id, github_id, embedding in issues:
        current_embedding = pickle.loads(embedding)
        current_duplicates = []

        for other_github_id, other_embedding in embeddings.items():
            if github_id == other_github_id:
                continue  # Skip self-comparison
            similarity = compute_cosine_similarity(current_embedding, other_embedding)
            if similarity >= DUPLICATE_THRESHOLD:
                current_duplicates.append({"issue_id": other_github_id, "similarity": similarity})

        duplicates[github_id] = current_duplicates

    update_duplicates(duplicates)