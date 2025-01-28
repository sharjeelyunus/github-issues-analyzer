# Analyzer Library

The `analyzer` library is a core module in the `github-issues-analyzer` monorepo. It orchestrates the fine-tuning of models, analysis of GitHub issues, and prediction of priorities and severities.

## Features

- **Fine-tune Models**: Fine-tunes a BERT-based model for label predictions based on GitHub issue datasets.
- **Fetch GitHub Issues**: Pulls issues from specified GitHub repositories.
- **Predict Priorities and Severities**: Uses zero-shot classification to predict priorities and severities for GitHub issues.
- **Find Duplicates**: Identifies potential duplicate issues in the dataset.
- **Assign Labels**: Automatically assigns labels to GitHub issues using the fine-tuned model.

## Installation

1. Clone the `github-issues-analyzer` monorepo:

   ```bash
   git clone https://github.com/your-username/github-issues-analyzer.git
   cd github-issues-analyzer

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To analyze issues and fine-tune the model:

Make sure the dataset is already generated using `generate_dataset.py`.
Run the analyze_issues function:

```bash
    python analyze_issues.py
```
