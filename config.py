import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.8))
LABELS_THRESHOLD = float(os.getenv("LABELS_THRESHOLD", 0.5))
DB_FILE = "issues.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
EMBEDDING_SIM_THRESHOLD= float(os.getenv("EMBEDDING_SIM_THRESHOLD", 0.70))
DATASET_REPOS_COUNT = int(os.getenv("DATASET_REPOS", 20))
LABELS_MODEL_DIR = "models/label_prediction_model_v1"
