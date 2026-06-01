import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import MNAddDataset
from models import DigitCNN

from utils import set_seed

def reasoning_loss_from_logits(digit1_logits, digit2_logits, target):
    digit1_probs = F.softmax(digit1_logits, dim=1)
    digit2_probs = F.softmax(digit2_logits, dim=1)

    d1 = torch.arange(10, device=digit1_probs.device)
    d2 = torch.arange(10, device=digit2_probs.device)

    pair_probs = digit1_probs.unsqueeze(2) * digit2_probs.unsqueeze(1)

    parity_mask = (d1.unsqueeze(1) + d2.unsqueeze(0)) % 2

    even_prob = pair_probs[:, parity_mask == 0].sum(dim=1)
    odd_prob = pair_probs[:, parity_mask == 1].sum(dim=1)

    reasoning_probs = torch.stack([even_prob, odd_prob], dim=1)
    reasoning_log_probs = torch.log(reasoning_probs + 1e-8)

    return F.nll_loss(reasoning_log_probs, target)


def train_combined(lambda_reasoning):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MNAddDataset(train=True)
    train_loader = DataLoader(train_dataset,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

    model = DigitCNN().to(device)

    concept_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(5):
        total_loss = 0

        for images, concepts, target in train_loader:
            images = images.to(device)
            concepts = concepts.to(device)
            target = target.to(device).long()

            digit1_labels = concepts[:, 0]
            digit2_labels = concepts[:, 1]

            optimizer.zero_grad()

            digit1_logits, digit2_logits = model(images)

            loss_digit1 = concept_criterion(digit1_logits, digit1_labels)
            loss_digit2 = concept_criterion(digit2_logits, digit2_labels)

            concept_loss = loss_digit1 + loss_digit2

            reasoning_loss = reasoning_loss_from_logits(
                digit1_logits, digit2_logits, target
            )

            loss = concept_loss + lambda_reasoning * reasoning_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Lambda {lambda_reasoning:.1f} | "
            f"Epoch {epoch + 1}, Loss: {total_loss:.4f}"
        )

    save_path = (
        f"../results/checkpoints/digit_cnn_combined_lambda_{lambda_reasoning:.1f}.pth"
    )
    torch.save(model.state_dict(), save_path)

    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    for lambda_reasoning in [0.1, 0.5, 1.0]:
        train_combined(lambda_reasoning)
