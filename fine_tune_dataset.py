from libs.analyzer.db_utils import load_dataset_for_analysis
from libs.analyzer.services.fine_tuning import (
    fine_tune_model,
    fine_tune_priority_severity_model,
)
from utils import extract_labels_from_issues


def fine_tune_on_dataset():
    """Fine-tune the model on dataset issues and save the fine-tuned model."""
    # Load dataset issues
    print("Loading dataset issues for fine-tuning...")
    dataset_issues = load_dataset_for_analysis()

    if not dataset_issues:
        print("No dataset issues found. Fine-tuning aborted.")
        return

    # Extract labels from issues
    print("Extracting labels from dataset issues...")
    labels = extract_labels_from_issues(dataset_issues)

    # Fine-tune the model
    print(f"Fine-tuning model on {len(dataset_issues)} dataset issues...")
    fine_tune_model(dataset_issues, labels)

    # Fine-tune priority and severity model
    print(
        f"Fine-tuning priority and severity model with {len(dataset_issues)} issues..."
    )
    fine_tune_priority_severity_model(dataset_issues)


if __name__ == "__main__":
    fine_tune_on_dataset()
