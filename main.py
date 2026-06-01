from src.dataset import MNAddDataset

dataset = MNAddDataset()

x, concepts, target = dataset[1]

print(x.shape)
print(concepts)
print(target)
