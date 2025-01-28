import sqlite3
import pickle
import json
from libs.dataset.dataset.database_manager import DatabaseManager
from utils import get_embedding

DB_FILE = "issues.db"


def initialize_db():
    """Initialize the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Drop the table if it exists
    cursor.execute("DROP TABLE IF EXISTS issues;")

    # Create the new table
    cursor.execute(
        """
        CREATE TABLE issues (
            id INTEGER PRIMARY KEY,
            github_id INTEGER UNIQUE,
            title TEXT,
            body TEXT,
            embedding BLOB,
            duplicates TEXT,
            labels TEXT,
            priority TEXT,
            severity TEXT
        );
    """
    )

    conn.commit()
    conn.close()


def store_issue(issue, embedding, priority=None, severity=None):
    """Store an issue in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Check if the issue already exists
    cursor.execute("SELECT id FROM issues WHERE github_id = ?", (issue["number"],))
    if cursor.fetchone():
        conn.close()
        return False

    # Serialize embedding using pickle
    serialized_embedding = pickle.dumps(embedding)

    existing_labels = json.dumps(issue.get("labels", []))

    # Insert into the database
    cursor.execute(
        """
        INSERT INTO issues (github_id, title, body, embedding, duplicates, labels, priority, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            issue["number"],
            issue["title"],
            issue.get("body", ""),
            serialized_embedding,
            "",
            existing_labels,
            priority,
            severity,
        ),
    )
    conn.commit()
    conn.close()
    return True


def fetch_all_issues():
    """Fetch all issues from the database, including labels."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, github_id, embedding, title, body, labels, priority, severity
        FROM issues
    """
    )
    issues = cursor.fetchall()
    conn.close()
    return issues


def update_duplicates(duplicates):
    """Update duplicates in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for github_id, duplicate_data in duplicates.items():
        cursor.execute(
            """
            UPDATE issues
            SET duplicates = ?
            WHERE github_id = ?
        """,
            (str(duplicate_data), github_id),
        )
    conn.commit()
    conn.close()


def store_issue_labels(github_id, labels):
    """Update labels for a specific issue."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE issues
        SET labels = ?
        WHERE github_id = ?
    """,
        (str(labels), github_id),
    )
    conn.commit()
    conn.close()


def update_priorities_and_severities(batch_results):
    """
    Update the priorities and severities for a batch of issues in the database.

    Args:
        batch_results: List of tuples with (github_id, priority, severity).
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for github_id, priority, severity in batch_results:
        cursor.execute(
            """
            UPDATE issues
            SET priority = ?, severity = ?
            WHERE github_id = ?
            """,
            (priority, severity, github_id),
        )

    conn.commit()
    conn.close()


def initialize_dataset_table():
    """Create the dataset table in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset (
            id INTEGER PRIMARY KEY,
            github_id INTEGER UNIQUE,
            title TEXT,
            body TEXT,
            embedding BLOB,
            duplicates TEXT,
            labels TEXT,
            priority TEXT,
            severity TEXT
        )
    """
    )
    conn.commit()
    conn.close()
    print("Table 'dataset' created or already exists.")


import json
import sqlite3
import pickle
from utils import get_embedding


def store_issues_in_dataset(repo_name, issues):
    """Store fetched issues in the dataset table."""
    conn = sqlite3.connect("issues.db")
    cursor = conn.cursor()

    for issue in issues:
        try:
            # Process labels into the required format
            raw_labels = issue.get("labels", [])
            labels = []

            if isinstance(raw_labels, list):
                # Convert list of strings into the desired object format
                for label in raw_labels:
                    labels.append(
                        {
                            "name": label,
                            "description": f"This issue is related to {label}",
                        }
                    )

            # Serialize labels to JSON for storage
            serialized_labels = json.dumps(labels)

            # Generate embeddings for the issue
            text = f"{issue['title']} {issue.get('body', '')}"
            embedding = pickle.dumps(get_embedding(text))  # Convert embedding to BLOB

            # Prepare placeholders for duplicates, priority, and severity
            duplicates = "[]"
            priority = None
            severity = None

            # Insert issue into the database
            cursor.execute(
                """
                INSERT OR IGNORE INTO dataset (
                    github_id, title, body, embedding, duplicates, labels, priority, severity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    issue["id"],
                    issue["title"],
                    issue.get("body", ""),
                    embedding,
                    duplicates,
                    serialized_labels,
                    priority,
                    severity,
                ),
            )
        except Exception as e:
            print(f"Failed to store issue {issue['id']}: {e}")

    conn.commit()
    conn.close()
    print(f"Stored {len(issues)} issues from {repo_name}.")


def load_dataset_for_analysis():
    """
    Load all dataset issues for analysis.
    """
    db_manager = DatabaseManager("issues.db")
    dataset_issues = db_manager.fetch_all_dataset_issues()
    db_manager.close()
    return dataset_issues
