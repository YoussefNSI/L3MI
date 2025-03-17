import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Chargement des données
with open("dataset_images_train", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
    X_train = data['data']
    y_train = data['target']

# Redimensionnement pour CNN (batch, channels, height, width)
X_train_tensor = th.tensor(X_train.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
y_train_tensor = th.tensor(y_train, dtype=th.long)

# Split train/validation
train_size = int(0.8 * len(X_train_tensor))
X_train_split, X_val_split = th.split(X_train_tensor, [train_size, len(X_train_tensor)-train_size])
y_train_split, y_val_split = th.split(y_train_tensor, [train_size, len(y_train_tensor)-train_size])

# Affichage d'une image
def show_image(img_tensor):
    plt.imshow(img_tensor.numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.show()

show_image(X_train_tensor[0])

# Visualisation t-SNE
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X_train_tensor[:3000].view(3000, -1).numpy())
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train[:3000], cmap='tab10', alpha=0.6)
plt.colorbar()
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Version optimisée avec sous-échantillonnage
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[:5000], y_train[:5000])
X_val_reshaped = X_val_split.numpy().reshape(-1, 3072)
y_pred = knn.predict(X_val_reshaped)
print("Accuracy KNN:", accuracy_score(y_val_split.numpy(), y_pred))
print("Accuracy KNN:", accuracy_score(y_val_split, y_pred))

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3072, 10)
    
    def forward(self, x):
        return self.linear(x.view(-1, 3072))

model = LogisticRegression()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entraînement
for epoch in range(50):
    outputs = model(X_train_split)
    loss = criterion(outputs, y_train_split)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entraînement avec DataLoader
use_full_data = True

if use_full_data:
    train_loader = DataLoader(TensorDataset(X_train_split, y_train_split), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_split, y_val_split), batch_size=64)
else:
    small_train_size = int(0.1 * len(X_train_split))
    small_val_size = int(0.1 * len(X_val_split))
    train_loader = DataLoader(TensorDataset(X_train_split[:small_train_size], y_train_split[:small_train_size]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_split[:small_val_size], y_val_split[:small_val_size]), batch_size=64)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with th.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total
    print(f"Epoch {epoch} - Loss: {loss.item():.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
    
# Chargement des données de test
# (À adapter selon la structure exacte des fichiers)
with open("data_images_test", 'rb') as fo:
    test_data = pickle.load(fo, encoding='bytes')['data']
X_test = th.tensor(test_data.reshape(-1, 3, 32, 32) / 255.0, dtype=th.float32)

# Prédiction avec le meilleur modèle
model.eval()
with th.no_grad():
    outputs = model(X_test)
    predictions = outputs.argmax(dim=1).numpy()

# Ensure predictions length matches the number of test samples
if len(predictions) < len(X_test):
    predictions = np.pad(predictions, (0, len(X_test) - len(predictions)), 'constant', constant_values=-1)

# Sauvegarde en CSV
pd.DataFrame(predictions.astype(float)).to_csv("images_test_predictions.csv", index=False)

