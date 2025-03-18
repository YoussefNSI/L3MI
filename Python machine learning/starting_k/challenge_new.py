import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from sklearn.manifold import TSNE
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

with open("dataset_images_train", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
    X_train = data['data']
    y_train = data['target']
    
# affichage de la première image

#redimensionnement 

X_train_tensor = th.tensor(X_train.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0) 
y_train_tensor = th.tensor(y_train, dtype=th.long)

def show_image(img_tensor):
    plt.imshow(img_tensor.numpy().transpose(1, 2, 0))
    plt.axis('off')
    plt.show()

#show_image(X_train_tensor[0])

def image_by_class(class_num):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    idx = (y_train_tensor == class_num).nonzero(as_tuple=True)[0]
    for i in range(10):
        img_tensor = X_train_tensor[idx[i]]
        axes[i].imshow(img_tensor.numpy().transpose(1, 2, 0))
        axes[i].axis('off')
    plt.show()

#image_by_class(5)

# Projection TSNE

"""
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_train_tensor[:3000].view(3000, -1).numpy())
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train[:3000], cmap='tab10', alpha=0.6)
plt.colorbar()
plt.show()
"""

# Modèle KNN

from sklearn.neighbors import KNeighborsClassifier

## Separation des données en train et test

X_train = X_train_tensor.view(X_train_tensor.size(0), -1).numpy()
y_train = y_train_tensor.numpy()

X_test = X_train_tensor[3000:4000].view(1000, -1).numpy()
y_test = y_train_tensor[3000:4000].numpy()

# Entrainement et évaluation du modèle pour différents k

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    accuracies.append(accuracy)

plt.plot(k_values, accuracies)
plt.title("Accuracy du KNN en fonction de k")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

test_knn = False

if test_knn:
    # Sélection du meilleur k
    best_k = k_values[np.argmax(accuracies)]
    print(f"Le meilleur k est {best_k} avec une accuracy de {max(accuracies):.2f}")

    # Application du KNN sur les données de test data_images_test avec le meilleur k

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)

    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        X_test = data['data']

    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
    X_test = X_test_tensor.view(X_test_tensor.size(0), -1).numpy()

    y_pred = knn.predict(X_test)

    # Sauvegarde des prédictions

    df = pd.DataFrame(y_pred, columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)

# Regression logistique multivariée

np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)

def prediction(f):
    return th.argmax(f, 1)

def error_rate(y_pred, y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]

## Frontière de décision

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def plot_decision(X, y, model):
    h = 0.05
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    
    x_flat = x_flat.reshape(x_flat.shape[0], 1)
    y_flat = y_flat.reshape(y_flat.shape[0], 1)
    
    X_entry = np.hstack((x_flat, y_flat))
    
    # Assurez-vous que X_entry a la bonne forme pour le modèle
    X_entry = th.from_numpy(X_entry).float().to(device)
    
    # Si votre modèle attend des données de dimension 3072, vous devez les transformer
    # Ici, nous supposons que vous avez déjà réduit les dimensions à 2 pour la visualisation
    # Assurez-vous que le modèle est compatible avec les dimensions d'entrée
    f = model(X_entry)
    y_pred = prediction(f).detach().cpu().numpy()
    preds = y_pred.reshape(xx.shape)
    
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, preds, cmap=cmap_light, shading='auto')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.show()
        

# Logistic Regression

X_train_split = X_train_tensor[:3000].view(3000, -1).to(device)
y_train_split = y_train_tensor[:3000].to(device)

X_val_split = X_train_tensor[3000:4000].view(1000, -1).to(device)
y_val_split = y_train_tensor[3000:4000].to(device)

test_regression_logistique = False

if test_regression_logistique:
    class LogisticRegression(nn.Module):
        def __init__(self, d, k):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(d, k)
        
        def forward(self, x):
            return self.linear(x)
        
    model = LogisticRegression(3072, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Entrainement


    for epoch in tqdm(range(500)):
        outputs = model(X_train_split)
        loss = criterion(outputs, y_train_split)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
        
    # Evaluation

    outputs = model(X_val_split)
    y_pred = prediction(outputs)
    error = error_rate(y_pred, y_val_split)
    print(f"Error rate: {error:.2f}")


    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        X_test = data['data']

    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
    X_test = X_test_tensor.view(X_test_tensor.size(0), -1).to(device)

    X_test_tensor = th.tensor(X_test, dtype=th.float32).to(device)
    outputs = model(X_test_tensor)
    y_pred = prediction(outputs)

    # Sauvegarde des prédictions

    df = pd.DataFrame(y_pred.cpu().numpy(), columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)
    
    # Frontière de décision

    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train[:3000])
    
    # Update the model to accept 2-dimensional input for plotting decision boundary
    model_pca = LogisticRegression(2, 10).to(device)
    model_pca.linear.weight.data = model.linear.weight.data[:, :2]
    model_pca.linear.bias.data = model.linear.bias.data

    plot_decision(X_train_pca, y_train[:3000], model_pca)
    

# Reseau de neurones avec des couches linéaires
test_neurone = False
if test_neurone:
    class NeuralNetwork(nn.Module):
        def __init__(self, d, k):
            super(NeuralNetwork, self).__init__()
            self.linear1 = nn.Linear(d, 100)
            self.linear2 = nn.Linear(100, k)
        
        def forward(self, x):
            x = F.relu(self.linear1(x))
            return self.linear2(x)
        
    model = NeuralNetwork(3072, 10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Entrainement

    for epoch in tqdm(range(500)):
        outputs = model(X_train_split)
        loss = criterion(outputs, y_train_split)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
        
    # Evaluation

    outputs = model(X_val_split)
    y_pred = prediction(outputs)
    error = error_rate(y_pred, y_val_split)

    print(f"Error rate: {error:.2f}")

    # Chargement des données de test
    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        X_test = data['data']

    # Préparation des données de test
    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
    X_test = X_test_tensor.view(X_test_tensor.size(0), -1).to(device)

    # Prédiction avec le modèle
    outputs = model(X_test)
    y_pred = prediction(outputs)

    # Sauvegarde des prédictions
    df = pd.DataFrame(y_pred.cpu().numpy(), columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)

test_CNN = True

if test_CNN:
    transform = transforms.Compose([
        ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])

    train_dataset = TensorDataset(X_train_tensor[:10000], y_train_tensor[:10000])
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=lambda batch: (
        th.stack([transform(img) for img, _ in batch]),
        th.stack([label for _, label in batch])
    ))
    val_loader = DataLoader(TensorDataset(X_train_tensor[9000:10000], y_train_tensor[9000:10000]), batch_size=128, shuffle=True)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.25)
            )
            self.fc_layers = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.fc_layers(x)
            return x

    # Utiliser ResNet-18 pré-entraîné
    class PretrainedCNN(nn.Module):
        def __init__(self):
            super(PretrainedCNN, self).__init__()
            self.model = resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
        def forward(self, x):
            return self.model(x)

    model = PretrainedCNN().to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    # Entraînement avec DataLoader et Mixed Precision Training
    early_stopping_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(50):
        model.train()
        for images, labels in train_loader:
            images = images.view(-1, 3, 32, 32).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with th.no_grad():
            for images, labels in val_loader:
                images = images.view(-1, 3, 32, 32).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch} - Loss: {val_loss:.4f} - Accuracy: {correct / total:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if epoch < 20:
                    continue
                print("Early stopping")
                break

    # Chargement des données de test
    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X_test = data['data']
        
    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0).to(device)
    
    # Prédiction avec le modèle
    model.eval()
    outputs = model(X_test_tensor)
    y_pred = prediction(outputs)
    
    # Sauvegarde des prédictions
    
    df = pd.DataFrame(y_pred.cpu().numpy(), columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)

