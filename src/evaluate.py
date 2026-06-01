import json
import torch
from torch.utils.data import DataLoader

from dataset import MNAddDataset
from models import DigitCNN, ConceptFilterMLP
from evaluator import Evaluator


def print_metrics(name, metrics):
    print(f"\n{name} RESULTS")
    print(f"Reasoning accuracy: {metrics['reasoning_accuracy']:.4f}")
    print(f"Concept accuracy: {metrics['concept_accuracy']:.4f}")
    print(f"Shortcut count: {metrics['shortcut_count']}")
    print(f"Shortcut frequency: {metrics['shortcut_frequency']:.4f}")
    print(
        f"Shortcut rate given correct reasoning: "
        f"{metrics['shortcut_rate_given_correct']:.4f}"
    )


def print_change_analysis(change_analysis):
    print("\nFILTER CHANGE ANALYSIS")
    print(f"Fixed cases: {change_analysis['fixed_cases']}")
    print(f"Broken cases: {change_analysis['broken_cases']}")
    print(f"Unchanged correct: {change_analysis['unchanged_correct']}")
    print(f"Unchanged wrong: {change_analysis['unchanged_wrong']}")
    print(f"Net gain: {change_analysis['net_gain']}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MNAddDataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Load trained CNN
    cnn = DigitCNN().to(device)
    cnn.load_state_dict(
        torch.load("../results/checkpoints/digit_cnn.pth", map_location=device)
    )

    # Baseline CNN evaluation
    baseline_evaluator = Evaluator(model=cnn, dataloader=test_loader, device=device)

    baseline_metrics = baseline_evaluator.evaluate_all()
    print_metrics("BASELINE CNN", baseline_metrics)

    # Load uncertainty-weighted residual filter
    filter_model = ConceptFilterMLP().to(device)
    filter_model.load_state_dict(
        torch.load(
            "../results/checkpoints/concept_filter_uncertainty_lambda_0.2.pth",
            map_location=device,
        )
    )

    # Filtered evaluation
    filtered_evaluator = Evaluator(
        model=cnn, filter_model=filter_model, dataloader=test_loader, device=device
    )

    filtered_metrics = filtered_evaluator.evaluate_all()
    print_metrics("UNCERTAINTY-WEIGHTED RESIDUAL FILTER λ=0.2", filtered_metrics)

    # Fixed / broken analysis
    change_analysis = filtered_evaluator.filter_change_analysis()
    print_change_analysis(change_analysis)

    # Save results
    all_metrics = {
        "baseline_cnn": baseline_metrics,
        "uncertainty_weighted_residual_filter_lambda_0.2": filtered_metrics,
        "filter_change_analysis": change_analysis,
    }

    save_path = "../results/metrics/uncertainty_filter_results.json"

    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nSaved results to {save_path}")


if __name__ == "__main__":
    main()