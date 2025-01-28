import subprocess
import analyze_issues
from config import API_HOST, API_PORT

# -----------------------------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------------------------
def main():
    # 1. Run the analysis (fetch GitHub issues and compute duplicates)
    print("Starting analysis...")
    analyze_issues.sync_issues()

    # 2. Start the API server
    print(f"Starting the API server at http://{API_HOST}:{API_PORT}...")
    subprocess.run(["uvicorn", "api:app", "--host", API_HOST, "--port", str(API_PORT), "--reload"])


if __name__ == "__main__":
    main()
