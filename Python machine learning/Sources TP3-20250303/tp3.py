import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



# Question 1

train_df = pd.read_csv("digits_train.csv")
print(train_df.head())

# Question 2

dim = train_df.shape[1] - 1
print("Dimension d = ", dim)

nb_classes = train_df.iloc[:, -1].nunique()
print("Nombre de classes = ", nb_classes)

# Question 3

first_line = train_df.iloc[0, :-1].values.reshape(8,8)

plt.figure()
plt.imshow(first_line, cmap='gray_r')
plt.title(f"Label : {train_df.iloc[0, -1]}")
plt.show()

# Question 4

train_df = train_df.sample(frac=1).reset_index(drop=True)
indice_split = int(0.6 * len(train_df))

digits_train = train_df.iloc[:indice_split]
digits_valid = train_df.iloc[indice_split:]

# Question 5

def dist_euclidienne(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(X_train, y_train, X_test, k=5):
    y_pred = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = dist_euclidienne(train_point, test_point)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_plus_proche = distances[:k]
        
        labels = [label for (_, label) in k_plus_proche]
        unique, counts = np.unique(labels, return_counts=True)
        y_pred.append(unique[np.argmax(counts)])
    
    return np.array(y_pred)

X_train = digits_train.iloc[:, :-1].values
y_train = digits_train.iloc[:, -1].values
X_valid = digits_valid.iloc[:, :-1].values
y_valid = digits_valid.iloc[:, -1].values

# Question 6

"""
k_values = list(range(1, 16))
accuracies = []

for k in k_values:
    y_pred_valid = knn(X_train, y_train, X_valid, k)
    acc = np.mean(y_pred_valid == y_valid)
    accuracies.append(acc)
    print(f"k={k} | Précision: {acc:.3f}")

best_k = k_values[np.argmax(accuracies)]
print(f"\nMeilleur k: {best_k} (Précision: {max(accuracies):.3f})")
"""

best_k = 1

# Question 7

test_entries = pd.read_csv("digits_entries_test.csv").values
test_target = pd.read_csv("digits_target_test.csv").values.flatten()

# Question 8

y_pred_test = knn(X_train, y_train, test_entries, best_k)
test_accuracy = np.mean(y_pred_test == test_target)
print(f"\nPrécision sur le test (k={best_k}): {test_accuracy:.3f}")

# Question 9

plt.figure(figsize=(12, 8))
for i in range(10):  # Afficher les 10 premiers
    plt.subplot(2, 5, i+1)
    digit = test_entries[i].reshape(8, 8)
    plt.imshow(digit, cmap='gray_r')
    plt.title(f"Prédit: {y_pred_test[i]}\nVrai: {test_target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Question 10 - 11 

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_df = pd.read_csv("digits_train.csv")
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

test_entries = pd.read_csv("digits_entries_test.csv").values
test_target = pd.read_csv("digits_target_test.csv").values.flatten()


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_entries)

class DigitClassifier(th.nn.Module):
    def __init__(self, d=64, k=10):
        super().__init__()
        self.layer = nn.Linear(d, k)
    
    def forward(self, x):
        return F.log_softmax(self.layer(x), dim=1)


model = DigitClassifier()
device = "cpu"
model = model.to(device)


X_train_tensor = th.tensor(X_train_scaled, dtype=th.float32).to(device)
y_train_tensor = th.tensor(y_train, dtype=th.long).to(device)
X_test_tensor = th.tensor(X_test_scaled, dtype=th.float32).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

nb_epochs = 20000
pbar = tqdm(range(nb_epochs))

for epoch in pbar:
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        with th.no_grad():
            preds = th.argmax(outputs, dim=1)
            acc = (preds == y_train_tensor).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=acc.item())


with th.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = th.argmax(test_outputs, dim=1).numpy()
    accuracy = np.mean(test_preds == test_target)
    print(f"\nAccuracy finale sur le test: {accuracy:.3f}")


plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    digit = test_entries[i].reshape(8, 8)
    plt.imshow(digit, cmap='gray_r')
    plt.title(f"Prédit: {test_preds[i]}\nVrai: {test_target[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()