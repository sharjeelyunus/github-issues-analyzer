import sqlite3
import json


class DatabaseManager:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        """
        Create a table to store GitHub issues if it doesn't already exist.
        """
        query = """
        CREATE TABLE IF NOT EXISTS dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            issue_id INTEGER UNIQUE NOT NULL,  
            repo TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            labels TEXT,
            priority TEXT,
            severity TEXT,
            state TEXT,
            created_at TEXT,
            updated_at TEXT,
            closed_at TEXT,
            author TEXT,
            assignees TEXT,
            num_comments INTEGER DEFAULT 0,
            num_reactions INTEGER DEFAULT 0,
            milestone TEXT,
            pull_request BOOLEAN DEFAULT 0
        )
        """
        self.cursor.execute(query)
        self.connection.commit()

    def insert_issues(self, issues):
        """
        Insert a list of issues into the database.
        """
        query = """
        INSERT OR IGNORE INTO dataset (
            issue_id, repo, title, body, labels, priority, severity, 
            state, created_at, updated_at, closed_at, author, 
            assignees, num_comments, num_reactions, milestone, pull_request
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                issue["issue_id"],
                issue["repo"],
                issue["title"],
                issue["body"],
                json.dumps(issue["labels"]),
                str(issue["priority"]),
                str(issue["severity"]),
                issue["state"],
                issue["created_at"],
                issue["updated_at"],
                issue["closed_at"],
                issue["author"],
                json.dumps(issue["assignees"]),
                issue["num_comments"],
                issue["num_reactions"],
                issue["milestone"],
                int(issue["pull_request"]),
            )
            for issue in issues
        ]
        self.cursor.executemany(query, data)
        self.connection.commit()

    def fetch_all_dataset_issues(self):
        """
        Fetch all issues from the dataset table.
        """
        query = """
        SELECT id, issue_id, repo, title, body, labels, priority, severity,
               state, created_at, updated_at, closed_at, author, assignees, 
               num_comments, num_reactions, milestone, pull_request
        FROM dataset
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        issues = []
        for row in rows:
            issues.append(
                {
                    "id": row[0],
                    "issue_id": row[1],
                    "repo": row[2],
                    "title": row[3],
                    "body": row[4],
                    "labels": json.loads(row[5]) if row[5] else [],  
                    "priority": row[6],
                    "severity": row[7],
                    "state": row[8],
                    "created_at": row[9],
                    "updated_at": row[10],
                    "closed_at": row[11],
                    "author": row[12],
                    "assignees": json.loads(row[13]) if row[13] else [],  
                    "num_comments": row[14],
                    "num_reactions": row[15],
                    "milestone": row[16],
                    "pull_request": bool(row[17]),
                }
            )
        return issues

    def close(self):
        """
        Close the database connection.
        """
        self.connection.close()
