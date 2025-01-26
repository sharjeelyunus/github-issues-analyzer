from sentence_transformers import SentenceTransformer, util

# Initialize model
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text: str):
    """Generate embedding for a given text."""
    return MODEL.encode(text).tolist()

def compute_cosine_similarity(vec_a, vec_b):
    """Compute the cosine similarity between two vectors."""
    return float(util.cos_sim(vec_a, vec_b)[0][0])
