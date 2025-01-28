# GitHub Issues Analyzer

A Python-based project to analyze GitHub issues using machine learning for semantic similarity. The project fetches open issues from a specified GitHub repository, analyzes them for duplicates using embeddings generated by a pre-trained Sentence Transformer model, assigns relevant labels based on issue context, predicts issue priority and severity, and stores the data in a local SQLite database.

---

## Features

- Fetch open issues from a GitHub repository using the GitHub API.
- Generate semantic embeddings for issue titles and descriptions.
- Identify and mark duplicate issues based on cosine similarity.
- Automatically assign relevant labels to issues based on their context during analysis.
- Predict priority (low, medium, high) and severity (minor, major, critical) for issues using machine learning models.
- Store issues, embeddings, labels, priority, and severity in a local SQLite database.
- Expose an API for accessing issues, duplicates, labels, priority, and severity.

---

## Requirements

- Python 3.10
- A GitHub Personal Access Token with the `repo` scope (for private repositories) or `public_repo` scope (for public repositories).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sharjeelyunus/github-issues-analyzer.git
cd github-issues-analyzer
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a .env file in the project root directory with the following content:

```plaintext
GITHUB_TOKEN=your_github_token
REPO_OWNER=your_repo_owner
REPO_NAME=your_repo_name
```

Replace your_github_token, your_repo_owner, and your_repo_name with your actual GitHub token and repository details.

### 5. Initialize the Database

The database will be automatically initialized when you run the script for the first time.

---

## Usage

### Generate Dataset

First, generate the dataset by running the `generate_dataset.py` script. This script fetches top github repository issues and stores them in the database. This dataset will be used for fine-tuning the model.

```bash
python generate_dataset.py
```

### Fine tune the model

Now, fine-tune the model by running the `fine_tune_dataset.py` script. This script will fine-tune the model on the dataset and save the model to the `models` directory.

```bash
python fine_tune_dataset.py
```

### Running the App

Run the `app.py` script to start the analyzer and API server:

```bash
python app.py
```

### Running the Analyzer

Run the `analyze_issues.py` script to fetch, analyze, and store issues:

```bash
python analyze_issues.py
```

This process dynamically fetches issue titles and descriptions from the database, ensures labels are assigned based on the most relevant and up-to-date context, and predicts priorities and severities for each issue.

### API Access

The project includes a FastAPI-based API for accessing issues, duplicates, labels, priority, and severity. Start the API server by running:

```bash
uvicorn api:app --reload
```

Access the API at <http://127.0.0.1:8000>. The API includes the following endpoints:

- GET /issues: List all issues with metadata.
- GET /issues/{github_id}: Get details of a specific issue, including duplicates, labels, priority, and severity.
- GET /duplicates: List issues that have potential duplicates.
- GET /labels: List issues with their assigned labels.
- GET /priorities-severities: List all issues with their predicted priority and severity.

---

## Examples

### Example API Response: /issues

```json
{
    "total": 10,
    "duplicates_count": 2,
    "labeled_issues_count": 5,
    "issues": [
        {
            "github_id": 1,
            "title": "Cannot save user",
            "body": "Error occurs when saving a new user",
            "duplicates": [
                {
                    "issue_id": 2,
                    "similarity": 82.0
                }
            ],
            "labels": ["bug", "backend"],
            "priority": "high",
            "severity": "critical"
        },
        {
            "github_id": 2,
            "title": "User save error",
            "body": "Fails with a database constraint violation",
            "duplicates": [],
            "labels": ["bug"],
            "priority": "medium",
            "severity": "major"
        }
    ]
}
```

---

## Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for pre-trained models used in semantic similarity and contextual labeling.
- [FastAPI](https://fastapi.tiangolo.com) for building the API.
- [GitHub API](https://docs.github.com/en/rest) for accessing issue data.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for zero-shot classification models, enabling contextual understanding for labels, priority, and severity predictions.
