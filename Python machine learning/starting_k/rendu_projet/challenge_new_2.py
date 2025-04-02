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
from PIL import Image
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import multiprocessing
import os
import copy
from torch.optim.swa_utils import AveragedModel, update_bn
from PIL import ImageOps, ImageEnhance

multiprocessing.freeze_support()
multiprocessing.set_start_method('spawn', force=True)


def apply_transforms(img_tensor, transform):
    img_pil = ToPILImage()(img_tensor)
    return transform(img_pil)

# Ajout d'une classe SimpleCNN adaptée aux images 32x32


with open("dataset_images_train", 'rb') as fo:
    data = pickle.load(fo, encoding='bytes')
    X_train = data['data']
    y_train = data['target']
    
# affichage de la première image

#redimensionnement 

X_train_tensor = th.tensor(X_train.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0) 
y_train_tensor = th.tensor(y_train, dtype=th.long)

mean = X_train_tensor.mean(dim=(0, 2, 3))  # [moyenne_R, moyenne_G, moyenne_B]
std = X_train_tensor.std(dim=(0, 2, 3))    # [std_R, std_G, std_B]

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

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = th.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

test_CNN = True

class SAM(th.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        # Initialisation correcte de l'optimiseur parent
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)
        
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = (th.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                
                # Sauvegarder l'état requires_grad et le désactiver temporairement
                requires_grad = p.requires_grad
                p.requires_grad_(False)
                
                # Effectuer l'opération in-place de manière sécurisée
                p.add_(e_w)
                
                # Restaurer l'état requires_grad
                p.requires_grad_(requires_grad)
                
                # Sauvegarder e_w pour la seconde étape
                self.state[p]["e_w"] = e_w

        if zero_grad: self.base_optimizer.zero_grad()

    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]: continue
                
                # Sauvegarder l'état requires_grad et le désactiver temporairement
                requires_grad = p.requires_grad
                p.requires_grad_(False)
                
                # Effectuer l'opération in-place de manière sécurisée
                p.sub_(self.state[p]["e_w"])
                
                # Restaurer l'état requires_grad
                p.requires_grad_(requires_grad)

        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        if zero_grad: self.base_optimizer.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM doesn't work like standard optimizers, use first_step and second_step.")

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)  # Déléguer à l'optimiseur de base

    def _grad_norm(self):
        norm = th.norm(
            th.stack([
                ((th.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # [32x32x32]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [16x16x32]
            nn.Conv2d(32, 64, 3, padding=1),  # [16x16x64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [8x8x64]
            nn.Conv2d(64, 128, 3, padding=1),  # [8x8x128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # [1x1x128]
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # [32x32x64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  # [32x32x64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [16x16x64]
            
            nn.Conv2d(64, 128, 3, padding=1),  # [16x16x128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),  # [16x16x128]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [8x8x128]
            
            nn.Conv2d(128, 256, 3, padding=1),  # [8x8x256]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),  # [8x8x256]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [4x4x256]
            
            nn.AdaptiveAvgPool2d(1)  # [1x1x256]
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def main():
    if test_CNN:
        # Utiliser une seed fixe pour la reproductibilité
        th.manual_seed(42)
        np.random.seed(42)
        
        print("Preprocessing training data...")
        
        # Remplacer RandAugment par des transformations plus douces
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ])
        
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist())
        ])
        
        # Prétraitement des données comme avant
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
        
        # Création des datasets
        train_dataset = TensorDataset(train_imgs, train_labels)
        val_dataset = TensorDataset(val_imgs, val_labels)
        
        # Data loaders avec stratégies d'optimisation
        train_loader = DataLoader(
            train_dataset, 
            batch_size=256,  # Batch size augmenté pour mieux exploiter le GPU
            shuffle=True,
            num_workers=6,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=512,
            shuffle=False,
            num_workers=6,
            pin_memory=True
        )
        
        # Créer un ensemble de différentes architectures
        models_configs = [
            {"name": "SimpleCNN", "model": SimpleCNN()},
            {"name": "DeepCNN", "model": DeepCNN()},
            {"name": "resnet18", "model": models.resnet18(pretrained=True)},
        ]
        
        trained_models = []
        
        # Pour chaque modèle de l'ensemble
        for model_config in models_configs:
            model_name = model_config["name"]
            model = model_config["model"]
            
            print(f"\nTraining {model_name}...")
            
            # Adapter les modèles pré-entraînés pour CIFAR-10
            if "resnet" in model_name:
                # Adapter correctement pour les petites images
                model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                model.maxpool = nn.Identity()  # Supprimer la couche MaxPool initiale
                num_ftrs = model.fc.in_features
                # Simplifier la tête de classification
                model.fc = nn.Sequential(
                    nn.BatchNorm1d(num_ftrs),
                    nn.Dropout(0.2),  # Dropout réduit
                    nn.Linear(num_ftrs, 10)
                )
                
            model = model.to(device)
            
            # S'assurer que tous les paramètres sont entraînables pour les modèles pré-entraînéss sont entraînables pour les modèles pré-entraînés
            for param in model.parameters():
                param.requires_grad = True
            
            model = model.to(device)
            
            # Vérifier et collecter explicitement les paramètres entraînables
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            
            # S'assurer qu'il y a des paramètres à entraîner
            if len(trainable_params) == 0:
                print(f"ERREUR: Le modèle {model_name} n'a pas de paramètres entraînables!")
                continue
                
            print(f"Nombre de paramètres entraînables pour {model_name}: {len(trainable_params)}")
            
            num_epochs = 300
            
            # Utiliser un optimiseur classique pour les modèles plus légers
            if model_name in ["SimpleCNN", "DeepCNN"]:
                optimizer = th.optim.SGD(
                    trainable_params,  # Utiliser les paramètres vérifiés
                    lr=0.001,
                    momentum=0.9,
                    weight_decay=5e-4,
                    nesterov=True
                )
            else:
                # Pour les modèles pré-entraînés, utiliser l'optimiseur SAM
                optimizer = SAM(
                    trainable_params,  # Utiliser les paramètres vérifiés au lieu de model.parameters()
                    th.optim.SGD,
                    lr=0.001,
                    momentum=0.9,
                    weight_decay=5e-4,
                    nesterov=True
                )
            
            # Scheduler OneCycleLR pour les modèles plus légers
            if model_name in ["SimpleCNN", "DeepCNN"]:
                scheduler = th.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=0.1,
                    steps_per_epoch=len(train_loader),
                    epochs=num_epochs,
                    pct_start=0.2,
                    div_factor=25.0,
                    final_div_factor=1000.0
                )
            else:
                # Pour SAM, accéder à l'optimiseur de base
                scheduler = th.optim.lr_scheduler.OneCycleLR(
                    optimizer.base_optimizer if isinstance(optimizer, SAM) else optimizer,
                    max_lr=0.05,
                    steps_per_epoch=len(train_loader),
                    epochs=num_epochs,
                    pct_start=0.2,
                    div_factor=25.0,
                    final_div_factor=1000.0
                )
            
            # Activer SWA après 75% de l'entraînement
            swa_start = int(0.75 * num_epochs)
            swa_model = AveragedModel(model)
            
            # Loss avec label smoothing modéré
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            # Tracking variables
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            
            # Early stopping avec patience accrue
            best_val_acc = 0.0
            best_model_state = None
            patience = 25  # Patience augmentée pour SWA
            patience_counter = 0
            
            # Training loop amélioré avec SAM et SWA
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for images, labels in tqdm(train_loader, desc=f"{model_name} - Epoch {epoch+1}/{num_epochs}"):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Réduire l'utilisation de Mixup/CutMix pour les petites images
                    # Mixup avec probabilité réduite
                    mixup_prob = 0.3 if epoch < swa_start else 0.1
                    mixup_alpha = 0.2  # Alpha réduit pour conserver plus d'information originale
                    
                    use_mixup = np.random.random() < mixup_prob
                    use_cutmix = False  # Désactiver CutMix qui peut trop altérer les petites images
                    
                    # Adapter la logique d'entraînement selon le type d'optimiseur
                    if isinstance(optimizer, SAM):
                        # Fonction de closure pour SAM
                        def closure():
                            if use_mixup:
                                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                                outputs = model(mixed_images)
                                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                            else:
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                            
                            loss.backward()
                            return loss, outputs
                        
                        # Premier pas de SAM
                        optimizer.zero_grad()
                        loss, outputs = closure()
                        optimizer.first_step(zero_grad=True)
                        
                        # Second pas de SAM
                        loss, outputs = closure()
                        optimizer.second_step(zero_grad=True)
                    else:
                        # Optimiseur standard (SGD)
                        optimizer.zero_grad()
                        
                        if use_mixup:
                            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                            outputs = model(mixed_images)
                            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        
                        loss.backward()
                        optimizer.step()
                    
                    # Mise à jour du scheduler
                    if scheduler is not None:
                        if isinstance(optimizer, SAM):
                            scheduler.step()
                        else:
                            scheduler.step()
                    
                    # Statistics
                    train_loss += loss.item() * images.size(0)
                    
                    if not use_mixup:
                        _, predicted = outputs.max(1)
                        train_total += labels.size(0)
                        train_correct += predicted.eq(labels).sum().item()
                
                # Activer SWA après le point de départ
                if epoch >= swa_start:
                    swa_model.update_parameters(model)
                
                # Calculer les métriques d'entraînement
                train_loss_epoch = train_loss / len(train_loader.dataset)
                train_acc_epoch = train_correct / max(train_total, 1)
                train_losses.append(train_loss_epoch)
                train_accuracies.append(train_acc_epoch)
                
                # Phase de validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with th.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * images.size(0)
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                val_loss_epoch = val_loss / len(val_loader.dataset)
                val_acc_epoch = val_correct / val_total
                val_losses.append(val_loss_epoch)
                val_accuracies.append(val_acc_epoch)
                
                # Afficher les métriques
                if isinstance(optimizer, SAM):
                    current_lr = optimizer.base_optimizer.param_groups[0]['lr']
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                print(f"{model_name} - Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, "
                      f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}, "
                      f"LR: {current_lr:.6f}")
                
                # Sauvegarder le meilleur modèle
                if val_acc_epoch > best_val_acc:
                    best_val_acc = val_acc_epoch
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    print(f"New best model! Val Acc: {val_acc_epoch:.4f}")
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{patience}")
                
                # Early stopping
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            
            # Mettre à jour les statistiques batch norm pour le modèle SWA
            if epoch >= swa_start:
                print("Updating SWA BatchNorm statistics...")
                with th.no_grad():
                    update_bn(train_loader, swa_model, device=device)
                
                # Évaluer le modèle SWA
                swa_model.eval()
                swa_val_correct = 0
                swa_val_total = 0
                
                with th.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        outputs = swa_model(images)
                        _, predicted = outputs.max(1)
                        swa_val_total += labels.size(0)
                        swa_val_correct += predicted.eq(labels).sum().item()
                
                swa_val_acc = swa_val_correct / swa_val_total
                print(f"SWA Validation Accuracy: {swa_val_acc:.4f}")
                
                # Utiliser le modèle SWA s'il est meilleur
                if swa_val_acc > best_val_acc:
                    best_val_acc = swa_val_acc
                    best_model_state = copy.deepcopy(swa_model.state_dict())
                    print(f"SWA model is better! Using it instead. Val Acc: {swa_val_acc:.4f}")
            
            # Sauvegarder le meilleur modèle
            model.load_state_dict(best_model_state)
            model_path = f"{model_name}_best_model.pth"
            th.save(model.state_dict(), model_path)
            trained_models.append({"name": model_name, "model": model, "val_acc": best_val_acc})
            
            print(f"Finished training {model_name}. Best validation accuracy: {best_val_acc:.4f}")
        
        # Sélection de plusieurs modèles pour l'ensemble
        additional_models_configs = [
            {"name": "resnet34", "model": models.resnet34(pretrained=True)},
            {"name": "efficientnet_b0", "model": models.efficientnet_b0(pretrained=True)},
            {"name": "densenet121", "model": models.densenet121(pretrained=True)},
            # Ajouter des variantes avec des hyperparamètres différents
            {"name": "resnet34_b", "model": models.resnet34(pretrained=True)},
        ]
        
        # Ne pas réinitialiser trained_models, ajoutez plutôt à la liste existante
        # trained_models = []  # Cette ligne est supprimée
        
        # Configuration optimisée pour chaque modèle
        for model_idx, model_config in enumerate(additional_models_configs):
           
            
            # Après le commentaire "Le reste du code d'entraînement reste le même...",
            # ajoutez le code d'entraînement complet ou simplement utilisez les modèles déjà entraînés
            
            # Par exemple, pour éviter d'entraîner ces modèles et utiliser uniquement les modèles déjà entraînés:
            print(f"Skipping training for {model_name} - will use existing models for ensemble")
            continue  # Passer à l'itération suivante
            
            # Ou si vous souhaitez utiliser le code d'entraînement pour ces modèles supplémentaires,
            # vous devez copier/coller le bloc d'entraînement ici
        
        

        # Vérifier que nous avons des modèles entraînés avant de faire des prédictions
        if len(trained_models) == 0:
            print("Aucun modèle entraîné n'est disponible pour les prédictions d'ensemble.")
            # Utiliser le premier modèle qui a été entraîné (SimpleCNN probablement)
            print("Utilisation du modèle SimpleCNN pour les prédictions...")
            # Charger le modèle SimpleCNN s'il existe sur le disque
            try:
                simple_model = SimpleCNN().to(device)
                simple_model.load_state_dict(th.load("SimpleCNN_best_model.pth"))
                trained_models.append({"name": "SimpleCNN", "model": simple_model, "val_acc": 1.0})  # val_acc fictive
            except:
                print("Impossible de charger le modèle SimpleCNN. Veuillez entraîner au moins un modèle.")
                return

        # Pseudo-labeling sur les données de test (technique avancée)
        # D'abord, prédire avec haute confiance en utilisant l'ensemble de modèles
        print("\nPerforming pseudo-labeling on test data...")
        
        # Charger les données de test
        with open("data_images_test", 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        X_test = data['data']
        
        X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
        
        # Normaliser les données de test
        X_test_normalized = []
        for img in X_test_tensor:
            normalized = transforms.Normalize(mean=mean.tolist(), std=std.tolist())(img)
            X_test_normalized.append(normalized)
        X_test_normalized = th.stack(X_test_normalized).to(device)
        
        # Prédictions d'ensemble pour le pseudo-labeling
        pseudo_labels = []
        pseudo_confidences = []
        
        batch_size = 256
        for i in tqdm(range(0, len(X_test_normalized), batch_size), desc="Ensemble predictions for pseudo-labeling"):
            batch = X_test_normalized[i:i+batch_size]
            ensemble_probs = []
            
            for model_info in trained_models:
                model = model_info["model"]
                model.eval()
                with th.no_grad():
                    outputs = model(batch)
                    probs = F.softmax(outputs, dim=1)
                    ensemble_probs.append(probs)
            
            # Moyenne des probabilités de l'ensemble
            avg_probs = th.stack(ensemble_probs).mean(dim=0)
            confidences, predictions = th.max(avg_probs, dim=1)
            
            pseudo_labels.extend(predictions.cpu().numpy())
            pseudo_confidences.extend(confidences.cpu().numpy())
        
        # Sélectionner les exemples à haute confiance pour le pseudo-labeling
        confidence_threshold = 0.95
        high_conf_indices = [i for i, conf in enumerate(pseudo_confidences) if conf > confidence_threshold]
        
        if high_conf_indices:
            print(f"Found {len(high_conf_indices)} high-confidence pseudo-labels (threshold: {confidence_threshold})")
            
            # Créer un dataset de pseudo-labels
            pseudo_images = X_test_normalized[high_conf_indices]
            pseudo_targets = th.tensor([pseudo_labels[i] for i in high_conf_indices], dtype=th.long).to(device)
            
            # Affiner le meilleur modèle avec les pseudo-labels
            best_model = trained_models[0]["model"]
            
            # Créer un petit dataloader pour les pseudo-labels
            pseudo_dataset = TensorDataset(pseudo_images, pseudo_targets)
            pseudo_loader = DataLoader(
                pseudo_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=2
            )
            
            print("Fine-tuning best model with pseudo-labels...")
            
            # Utiliser un faible taux d'apprentissage
            fine_tune_optimizer = optim.SGD(best_model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
            
            # Ré-entraîner brièvement avec des pseudo-labels
            best_model.train()
            for epoch in range(5):  # Quelques époques seulement
                for images, labels in tqdm(pseudo_loader, desc=f"Pseudo-labeling epoch {epoch+1}/5"):
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    fine_tune_optimizer.zero_grad()
                    outputs = best_model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    fine_tune_optimizer.step()
            
            print("Fine-tuning with pseudo-labels completed")
        
        # Utiliser l'ensemble des modèles entraînés pour les prédictions finales
        print("\nGenerating final predictions with ensemble...")
        
        # Prédictions d'ensemble
        final_predictions = []
        
        for batch_idx in tqdm(range(0, len(X_test_normalized), batch_size), desc="Final ensemble predictions"):
            batch = X_test_normalized[batch_idx:batch_idx+batch_size]
            ensemble_probs = []
            
            # Obtenir les prédictions de chaque modèle
            for model_info in trained_models:
                model = model_info["model"]
                model.eval()
                
                # Test-time augmentation pour chaque modèle
                batch_probs = []
                
                # Prédiction originale
                with th.no_grad():
                    outputs = model(batch)
                    probs = F.softmax(outputs, dim=1)
                    batch_probs.append(probs)
                
                # Prédiction avec flip horizontal
                with th.no_grad():
                    flipped = th.flip(batch, dims=[3])  # Flip horizontal
                    outputs = model(flipped)
                    probs = F.softmax(outputs, dim=1)
                    batch_probs.append(probs)
                
                # Moyenne des prédictions TTA pour ce modèle
                avg_model_probs = th.stack(batch_probs).mean(dim=0)
                ensemble_probs.append(avg_model_probs)
            
            # Moyenne pondérée par la précision de validation
            weighted_probs = th.zeros_like(ensemble_probs[0])
            total_weight = 0
            
            for model_idx, probs in enumerate(ensemble_probs):
                weight = trained_models[model_idx]["val_acc"]
                weighted_probs += probs * weight
                total_weight += weight
            
            weighted_probs /= total_weight
            
            # Obtenir les classes prédites
            _, predictions = th.max(weighted_probs, dim=1)
            final_predictions.extend(predictions.cpu().numpy())
        
        # Sauvegarder les prédictions finales
        df = pd.DataFrame(final_predictions, columns=["target"])
        df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
        df.to_csv("images_test_predictions.csv", index=False, header=False)
        
        print("Ensemble predictions saved to images_test_predictions.csv")

if __name__ == "__main__":
    main()

# Fonction auxiliaire pour CutMix
def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



