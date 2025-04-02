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
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
# Importation du multiprocessing mais sans l'initialiser ici
import multiprocessing
import os

def apply_transforms(img_tensor, transform):
    img_pil = ToPILImage()(img_tensor)
    return transform(img_pil)

# Définir les transformations une seule fois pour éviter de les redéfinir dans collate_fn
train_transform = transforms.Compose([
    ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

# Définir une fonction collate_fn en dehors de la fonction main pour qu'elle soit picklable
def collate_fn_transforms(batch):
    images = [apply_transforms(img, train_transform) for img, _ in batch]
    labels = [label for _, label in batch]
    return th.stack(images), th.stack(labels)

# Remplacer get_collate_fn par une classe TransformCollate
class TransformCollate:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, batch):
        imgs = [self.transform(img) for img, _ in batch]
        labels = [label for _, label in batch]
        return th.stack(imgs), th.stack(labels)

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
    
    X_entry = th.from_numpy(X_entry).float().to(device)
    
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


def main():
    if test_CNN:
        print("Preprocessing training data...")

        # Calculer la moyenne et l'écart-type pour la normalisation
        mean = X_train_tensor.mean(dim=(0, 2, 3))
        std = X_train_tensor.std(dim=(0, 2, 3))
        print(f"Statistiques de normalisation - Mean: {mean}, Std: {std}")

        # Supprimer ToPILImage() du début car apply_transforms convertit déjà en PIL
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        # Prétraitement des données avec application directe des transformations
        train_imgs = []
        train_labels = []
        
        for i in tqdm(range(len(X_train_tensor[:9000])), desc="Preprocessing training data"):
            img = X_train_tensor[i]
            label = y_train_tensor[i]
            img_transformed = apply_transforms(img, transform_train)
            train_imgs.append(img_transformed)
            train_labels.append(label)
        
        train_imgs = th.stack(train_imgs)
        train_labels = th.stack(train_labels)
        
        val_imgs = []
        val_labels = []
        
        for i in tqdm(range(len(X_train_tensor[9000:10000])), desc="Preprocessing validation data"):
            img = X_train_tensor[9000 + i]
            label = y_train_tensor[9000 + i]
            img_transformed = apply_transforms(img, transform_val)
            val_imgs.append(img_transformed)
            val_labels.append(label)
        
        val_imgs = th.stack(val_imgs)
        val_labels = th.stack(val_labels)
        
        # Création des datasets avec les données prétraitées
        train_dataset = TensorDataset(train_imgs, train_labels)
        val_dataset = TensorDataset(val_imgs, val_labels)
        
        # DataLoader sans collate_fn (car prétraitement déjà appliqué)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=256, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=256, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

        # Modèle PretrainedCNN basé sur resnet18 pré-entraîné
        class PretrainedCNN(nn.Module):
            def __init__(self):
                super(PretrainedCNN, self).__init__()
                self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.model.fc = nn.Linear(self.model.fc.in_features, 10)
            def forward(self, x):
                return self.model(x)
                
        model = PretrainedCNN().to(device)
        model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = GradScaler('cuda' if th.cuda.is_available() else 'cpu')

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        early_stopping_patience = 5
        best_val_loss = float('inf')
        patience_counter = 0

        epoch_bar = tqdm(range(100), desc="Training", position=0)
        for epoch in epoch_bar:
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            for images, labels in batch_bar:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with autocast('cuda' if th.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                batch_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            train_loss_epoch = train_loss / train_total
            train_acc_epoch = train_correct / train_total
            train_losses.append(train_loss_epoch)
            train_accuracies.append(train_acc_epoch)

            # Phase de validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with th.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_loss_epoch = val_loss / total
            val_acc_epoch = correct / total
            val_losses.append(val_loss_epoch)
            val_accuracies.append(val_acc_epoch)

            scheduler.step(val_loss_epoch)
            epoch_bar.set_postfix({
                "Train Loss": f"{train_loss_epoch:.4f}",
                "Train Acc": f"{train_acc_epoch:.4f}",
                "Val Loss": f"{val_loss_epoch:.4f}",
                "Val Acc": f"{val_acc_epoch:.4f}",
                "LR": f"{scheduler.get_last_lr()[0]:.6f}"
            })

            if val_loss_epoch < best_val_loss:
                best_val_loss = val_loss_epoch
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience and epoch >= 20:
                    print("\nEarly stopping triggered")
                    break

        # Préparation et prédiction sur les données de test
        with open("data_images_test", 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            X_test = data['data']
        X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
        
        # Prétraitement des données de test
        test_imgs = []
        for i in tqdm(range(len(X_test_tensor)), desc="Preprocessing test data"):
            img = X_test_tensor[i]
            img_transformed = apply_transforms(img, transform_val)
            test_imgs.append(img_transformed)
        
        test_imgs = th.stack(test_imgs).to(device)
        
        # Prédiction avec le modèle
        model.eval()
        all_preds = []
        
        batch_size = 256
        for i in range(0, len(test_imgs), batch_size):
            with th.no_grad():
                outputs = model(test_imgs[i:i+batch_size])
                preds = th.argmax(outputs, 1)
                all_preds.append(preds)
        
        y_pred = th.cat(all_preds).cpu()

        df = pd.DataFrame(y_pred.numpy(), columns=["target"])
        df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
        df.to_csv("images_test_predictions.csv", index=False, header=False)

        # Affichage des courbes d'entraînement
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='orange', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy', color='green')
        plt.plot(val_accuracies, label='Val Accuracy', color='red', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()


