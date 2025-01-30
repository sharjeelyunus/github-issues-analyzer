import sqlite3
import pandas as pd

from config import DB_FILE

def convert_to_parquet(sqlite_path=DB_FILE, table_name="dataset", path="github_issues_dataset.parquet"):
    """
    Convert an SQLite table to a Parquet file for efficient storage & ML training.
    Args:
        sqlite_path (str): Path to the SQLite database.
        table_name (str): Table name to export.
        path (str): Destination Parquet file.
    """
    try:
        conn = sqlite3.connect(sqlite_path)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Save in Parquet format
        df.to_parquet(path, index=False)
        print(f"✅ Dataset exported to {path}")
        return path
    except Exception as e:
        print(f"❌ Error exporting dataset: {e}")
        return None


if __name__ == "__main__":
    print("🔄 Converting dataset to CSV...")
    parquet_file = convert_to_parquet()

    if parquet_file:
        print(f"✅ CSV generated: {parquet_file}")
    else:
        print("❌ Failed to generate CSV.")