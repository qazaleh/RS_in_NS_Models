import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MNAddDataset
from models import DigitCNN, ConceptFilterMLP
from utils import set_seed


def train_filter(lambda_uncertainty=0.2):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MNAddDataset(train=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    cnn = DigitCNN().to(device)
    cnn.load_state_dict(
        torch.load("../results/checkpoints/digit_cnn.pth", map_location=device)
    )

    for param in cnn.parameters():
        param.requires_grad = False

    cnn.eval()

    filter_model = ConceptFilterMLP().to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(filter_model.parameters(), lr=0.001, weight_decay=1e-4)

    filter_model.train()

    for epoch in range(5):
        total_loss = 0

        for images, concepts, target in train_loader:
            images = images.to(device)
            concepts = concepts.to(device)

            digit1_labels = concepts[:, 0]
            digit2_labels = concepts[:, 1]

            optimizer.zero_grad()

            with torch.no_grad():
                digit1_logits, digit2_logits = cnn(images)

                digit1_probs = F.softmax(digit1_logits, dim=1)
                digit2_probs = F.softmax(digit2_logits, dim=1)

                concept_probs = torch.cat([digit1_probs, digit2_probs], dim=1)
                original_logits = torch.cat([digit1_logits, digit2_logits], dim=1)

            filtered_digit1_logits, filtered_digit2_logits = filter_model(
                concept_probs, original_logits
            )

            loss1 = criterion(filtered_digit1_logits, digit1_labels)
            loss2 = criterion(filtered_digit2_logits, digit2_labels)

            sample_loss = loss1 + loss2

            filtered_digit1_probs = F.softmax(filtered_digit1_logits, dim=1)
            filtered_digit2_probs = F.softmax(filtered_digit2_logits, dim=1)

            digit1_conf = filtered_digit1_probs.max(dim=1).values
            digit2_conf = filtered_digit2_probs.max(dim=1).values

            uncertainty = 2.0 - (digit1_conf + digit2_conf)

            weights = 1.0 + lambda_uncertainty * uncertainty

            loss = (sample_loss * weights).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, " f"Filter Loss: {total_loss:.4f}")

    save_path = (
        f"../results/checkpoints/"
        f"concept_filter_uncertainty_lambda_{lambda_uncertainty:.1f}.pth"
    )

    torch.save(filter_model.state_dict(), save_path)

    print(f"Saved filter model to {save_path}")


if __name__ == "__main__":
    train_filter(lambda_uncertainty=0.2)
