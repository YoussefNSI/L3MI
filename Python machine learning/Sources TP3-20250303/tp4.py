from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch as th

data = fetch_olivetti_faces()
X = data.data
y = data.target


# Question 2

dim = data.images.shape[1] * data.images.shape[2]
print("Dimension d = ", dim)
classes = len(set(data.target))
print("Nombre de classes = ", classes)

# Question 3

plt.imshow(X[0].reshape(64, 64), cmap='gray')
plt.axis('off')
plt.show()

# Question 4

def afficher_classe(classe):
    indices = np.where(y == classe)[0]
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, 10, i+1)
        plt.imshow(X[idx].reshape(64, 64), cmap='gray')
        plt.axis('off')
    plt.show()

afficher_classe(0)

# Question 5

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Question 6

class ModeleLineaire(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.fc = nn.Linear(d, k)
    
    def forward(self, x):
        return self.fc(x)

# Question 7

X_train_t = th.tensor(X_train, dtype=th.float32)
y_train_t = th.tensor(y_train, dtype=th.long)
X_val_t = th.tensor(X_val, dtype=th.float32)
y_val_t = th.tensor(y_val, dtype=th.long)

model = ModeleLineaire(d=4096, k=40)
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5000):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        with th.no_grad():
            train_acc = (outputs.argmax(1) == y_train_t).float().mean().item()
            val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()
            print(f"Epoch {epoch} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

# Question 8

class ReseauNeurones(nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(h, k)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# Question 9

model = ReseauNeurones(d=4096, h=50, k=40)
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

# Question 11

hidden_sizes = [1, 5, 10, 50, 100]
val_errors = []

for h in hidden_sizes:
    model = ReseauNeurones(4096, h, 40)
    val_preds = model(X_val_t).argmax(1)
    val_error = 1 - (val_preds == y_val_t).float().mean().item()
    val_errors.append(val_error)

plt.plot(hidden_sizes, val_errors)
plt.xlabel("Nombre de neurones")
plt.ylabel("Erreur de validation")
plt.show()

# Question 12

class ReseauProfond(nn.Module):
    def __init__(self, d, h1, h2, k):
        super().__init__()
        self.fc1 = nn.Linear(d, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, k)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Question 13

with th.no_grad():
    val_preds = model(X_val_t).argmax(1)

erreurs = (val_preds != y_val_t).nonzero().squeeze()

plt.figure(figsize=(15, 10))
for i, idx in enumerate(erreurs):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_val[idx].reshape(64, 64), cmap='gray')
    plt.title(f"Prédit: {val_preds[idx]}\nVrai: {y_val[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()



