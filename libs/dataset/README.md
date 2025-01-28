# Dataset Library

The `dataset` library is part of the `github-issues-analyzer` monorepo. It fetches the top GitHub repositories' issues and stores them in an SQLite database

## Features

- Fetches the top repositories from GitHub.
- Fetches all open issues for each repository.
- Stores the issues in an SQLite database.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sharjeelyunus/github-issues-analyzer.git
   cd github-issues-analyzer/libs/dataset
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a .env file in the root directory:

    ```bash
    GITHUB_TOKEN=your_personal_access_token
    ```

## Usage

Run the generate_dataset script to fetch and store GitHub issues:

```bash
python generate_dataset.py
```
