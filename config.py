import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = os.getenv("REPO_OWNER")
REPO_NAME = os.getenv("REPO_NAME")
DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", 0.8))
DB_FILE = "issues.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
