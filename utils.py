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