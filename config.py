import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_TOKENS = os.getenv("GITHUB_TOKENS").split(",") if os.getenv("GITHUB_TOKENS") else [GITHUB_TOKEN]
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
DUPLICATE_THRESHOLD = 0.8
LABELS_THRESHOLD = 0.5
DB_FILE = "issues.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

LABELS_MODEL_DIR = "models/label_prediction_model_v1"
PRIORITY_SEVERITY_MODEL_DIR = "models/priority_severity_model_v1"
DATASET_REPOS_COUNT = 200

API_HOST = "127.0.0.1"
API_PORT = 8000
