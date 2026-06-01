import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,16, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(16,32, kernel_size = 3, padding = 1)

        self.pool = nn.MaxPool2d(2,2)   

        self.fc1 = nn.Linear(32 * 7 * 14 , 128)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # 1 * 28 * 56 --> 16 * 14 * 28
        x = self.pool(F.relu(self.conv2(x))) # 16 * 14 * 28 --> 32 * 7 * 14 

        x = x.view(x.size(0), -1) 

        x = F.relu(self.fc1(x)) 

        logits = self.fc2(x)

        digit1_logits = logits[:, :10]
        digit2_logits = logits[:, 10:]

        return digit1_logits, digit2_logits


class ConceptFilterMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.correction_net = nn.Sequential(
            nn.Linear(20, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 20)
        )

        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)

    def forward(self, concept_probs, original_logits):
        correction = self.correction_net(concept_probs)

        filtered_logits = original_logits + correction

        digit1_logits = filtered_logits[:, :10]
        digit2_logits = filtered_logits[:, 10:]

        return digit1_logits, digit2_logits
