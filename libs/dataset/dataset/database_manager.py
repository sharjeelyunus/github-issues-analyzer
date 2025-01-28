import sqlite3


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
            id INTEGER PRIMARY KEY,
            repo TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT,
            labels TEXT,
            priority TEXT,
            severity TEXT
        )
        """
        self.cursor.execute(query)
        self.connection.commit()

    def insert_issues(self, issues):
        """
        Insert a list of issues into the database.
        """
        query = """
        INSERT OR IGNORE INTO dataset (id, repo, title, body, labels, priority, severity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        data = [
            (
                issue["id"],
                issue["repo"],
                issue["title"],
                issue["body"],
                ",".join(issue["labels"]),
                str(issue["priority"]),
                str(issue["severity"]),
            )
            for issue in issues
        ]
        self.cursor.executemany(query, data)
        self.connection.commit()

    def close(self):
        """
        Close the database connection.
        """
        self.connection.close()
