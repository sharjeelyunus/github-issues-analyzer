import sqlite3
import pickle

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
            labels TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_issue(issue, embedding):
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

    # Insert into the database
    cursor.execute("""
        INSERT INTO issues (github_id, title, body, embedding, duplicates, labels)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        issue["number"],
        issue["title"],
        issue.get("body", ""),
        serialized_embedding,
        "",
        ""
    ))
    conn.commit()
    conn.close()
    return True

def fetch_all_issues_with_embeddings():
    """Fetch all issues with embeddings from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, github_id, embedding FROM issues")
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
