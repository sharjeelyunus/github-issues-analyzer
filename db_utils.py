import sqlite3
import pickle
from typing import List, Tuple

DB_FILE = "issues.db"

def initialize_db():
    """Initialize the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
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
    """)
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

    # Store existing labels as a JSON array
    import json
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
            severity
        )
    )
    conn.commit()
    conn.close()
    return True

def fetch_all_issues():
    """Fetch all issues from the database, including labels."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, github_id, embedding, title, body, labels, priority, severity
        FROM issues
    """)
    issues = cursor.fetchall()
    conn.close()
    return issues

def update_duplicates(duplicates):
    """Update duplicates in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for github_id, duplicate_data in duplicates.items():
        cursor.execute("""
            UPDATE issues
            SET duplicates = ?
            WHERE github_id = ?
        """, (str(duplicate_data), github_id))
    conn.commit()
    conn.close()

def store_issue_labels(github_id, labels):
    """Update labels for a specific issue."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE issues
        SET labels = ?
        WHERE github_id = ?
    """, (str(labels), github_id))
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
