import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MNAddDataset
from models import DigitCNN, ConceptFilterMLP
from reasoning import predict_digits


def analyze_shortcuts(max_examples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = MNAddDataset(train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    cnn = DigitCNN().to(device)
    cnn.load_state_dict(
        torch.load("../results/checkpoints/digit_cnn.pth", map_location=device)
    )
    cnn.eval()

    filter_model = ConceptFilterMLP().to(device)
    filter_model.load_state_dict(
        torch.load(
            "../results/checkpoints/concept_filter_uncertainty_lambda_0.2.pth",
            map_location=device,
        )
    )
    filter_model.eval()

    rows = []
    sample_index = 0

    with torch.no_grad():
        for images, concepts, target in test_loader:
            images = images.to(device)
            concepts = concepts.to(device)
            target = target.to(device)

            digit1_true = concepts[:, 0]
            digit2_true = concepts[:, 1]

            # CNN outputs
            cnn_digit1_logits, cnn_digit2_logits = cnn(images)

            cnn_digit1_probs = F.softmax(cnn_digit1_logits, dim=1)
            cnn_digit2_probs = F.softmax(cnn_digit2_logits, dim=1)

            cnn_digit1_pred, cnn_digit2_pred = predict_digits(
                cnn_digit1_logits, cnn_digit2_logits
            )

            cnn_reasoning_pred = (cnn_digit1_pred + cnn_digit2_pred) % 2

            cnn_reasoning_correct = cnn_reasoning_pred == target
            cnn_concepts_correct = (cnn_digit1_pred == digit1_true) & (
                cnn_digit2_pred == digit2_true
            )

            cnn_shortcut_cases = cnn_reasoning_correct & (~cnn_concepts_correct)

            # Filter outputs
            concept_probs = torch.cat([cnn_digit1_probs, cnn_digit2_probs], dim=1)

            original_logits = torch.cat([cnn_digit1_logits, cnn_digit2_logits], dim=1)

            filtered_digit1_logits, filtered_digit2_logits = filter_model(
                concept_probs, original_logits
            )

            filtered_digit1_probs = F.softmax(filtered_digit1_logits, dim=1)
            filtered_digit2_probs = F.softmax(filtered_digit2_logits, dim=1)

            filtered_digit1_pred, filtered_digit2_pred = predict_digits(
                filtered_digit1_logits, filtered_digit2_logits
            )

            filtered_reasoning_pred = (filtered_digit1_pred + filtered_digit2_pred) % 2

            for i in range(images.size(0)):
                if not cnn_shortcut_cases[i]:
                    sample_index += 1
                    continue

                true_d1 = digit1_true[i].item()
                true_d2 = digit2_true[i].item()

                cnn_pred_d1 = cnn_digit1_pred[i].item()
                cnn_pred_d2 = cnn_digit2_pred[i].item()

                filtered_pred_d1 = filtered_digit1_pred[i].item()
                filtered_pred_d2 = filtered_digit2_pred[i].item()

                filter_fixed = (
                    filtered_pred_d1 == true_d1 and filtered_pred_d2 == true_d2
                )

                rows.append(
                    {
                        "sample_index": sample_index,
                        "true_digit1": true_d1,
                        "true_digit2": true_d2,
                        "cnn_pred_digit1": cnn_pred_d1,
                        "cnn_pred_digit2": cnn_pred_d2,
                        "filtered_pred_digit1": filtered_pred_d1,
                        "filtered_pred_digit2": filtered_pred_d2,
                        "true_parity": target[i].item(),
                        "cnn_parity": cnn_reasoning_pred[i].item(),
                        "filtered_parity": filtered_reasoning_pred[i].item(),
                        "cnn_digit1_true_prob": cnn_digit1_probs[i, true_d1].item(),
                        "cnn_digit1_pred_prob": cnn_digit1_probs[i, cnn_pred_d1].item(),
                        "cnn_digit2_true_prob": cnn_digit2_probs[i, true_d2].item(),
                        "cnn_digit2_pred_prob": cnn_digit2_probs[i, cnn_pred_d2].item(),
                        "filtered_digit1_true_prob": filtered_digit1_probs[
                            i, true_d1
                        ].item(),
                        "filtered_digit1_pred_prob": filtered_digit1_probs[
                            i, filtered_pred_d1
                        ].item(),
                        "filtered_digit2_true_prob": filtered_digit2_probs[
                            i, true_d2
                        ].item(),
                        "filtered_digit2_pred_prob": filtered_digit2_probs[
                            i, filtered_pred_d2
                        ].item(),
                        "filter_fixed_shortcut": filter_fixed,
                    }
                )

                sample_index += 1

                if max_examples is not None and len(rows) >= max_examples:
                    break

            if max_examples is not None and len(rows) >= max_examples:
                break

    save_path = "../results/metrics/shortcut_analysis.csv"

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} shortcut cases to {save_path}")


if __name__ == "__main__":
    analyze_shortcuts()