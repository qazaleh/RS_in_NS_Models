import torch
import torch.nn as nn 
from torch.utils.data import DataLoader

from dataset import MNAddDataset
from models import DigitCNN

from utils import set_seed
"""
image counts = 60000
batch size = 64
60000 / 64 ≈ 937 batch counts
937 forward 
937 backward
"""

def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MNAddDataset(train=True)
    train_loader = DataLoader(train_dataset,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

    model = DigitCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(5):
        total_loss = 0 

        for images, concepts, target in train_loader:
            images = images.to(device)
            concepts = concepts.to(device)

            digit1_labels = concepts[:, 0]
            digit2_labels = concepts[:, 1]

            optimizer.zero_grad()

            digit1_logits, digit2_logits = model(images)

            loss1 = criterion(digit1_logits, digit1_labels)
            loss2 = criterion(digit2_logits, digit2_labels)

            # Train DigitCNN using only concept supervision
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "../results/checkpoints/digit_cnn.pth")


if __name__ == "__main__":
    train()
