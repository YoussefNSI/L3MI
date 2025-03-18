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

test_CNN = True

if test_CNN:
    # Improved data augmentation and normalization
    transform_train = transforms.Compose([
        ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    transform_val = transforms.Compose([
        ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # Using more data for training, dedicated validation set
    train_dataset = TensorDataset(X_train_tensor[:9000], y_train_tensor[:9000])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=lambda batch: (
        th.stack([transform_train(img) for img, _ in batch]),
        th.stack([label for _, label in batch])
    ))
    
    val_dataset = TensorDataset(X_train_tensor[9000:10000], y_train_tensor[9000:10000])
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=lambda batch: (
        th.stack([transform_val(img) for img, _ in batch]),
        th.stack([label for _, label in batch])
    ))

    # Basic residual block for better gradient flow
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # Skip connection with projection if needed
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    # Improved CNN model with residual connections
    class ImprovedCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(ImprovedCNN, self).__init__()
            
            # Initial layers
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            
            # Residual layers
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            
            # Final classifier
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, num_classes)
            
            # Dropout for regularization
            self.dropout = nn.Dropout(0.2)
            
        def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
            layers = []
            layers.append(ResidualBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResidualBlock(out_channels, out_channels))
            return nn.Sequential(*layers)
            
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.fc(out)
            
            return out

    # Initialize model, optimizer, and loss function
    model = ImprovedCNN().to(device)
    if th.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Weight initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Using One Cycle Learning Rate policy
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01, 
        steps_per_epoch=len(train_loader), 
        epochs=50,
        pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()

    # Training metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Early stopping parameters
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    max_epochs = 50

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss_epoch = train_loss / train_total
        train_acc_epoch = train_correct / train_total
        train_losses.append(train_loss_epoch)
        train_accuracies.append(train_acc_epoch)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with th.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss_epoch = val_loss / val_total
        val_acc_epoch = val_correct / val_total
        val_losses.append(val_loss_epoch)
        val_accuracies.append(val_acc_epoch)
        
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")
        
        # Save the best model
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            best_model = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model saved! Validation loss: {val_loss_epoch:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
        # Early stopping
        if patience_counter >= patience and epoch > 20:
            print("Early stopping triggered")
            break
    
    # Load best model for testing
    model.load_state_dict(best_model)
    
    # Test predictions
    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X_test = data['data']
        
    # Process test data with the same normalization
    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
    test_dataset = TensorDataset(X_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=lambda batch: (
        th.stack([transform_val(img[0]) for img in batch]),
    ))
    
    # Make predictions
    model.eval()
    all_predictions = []
    
    with th.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
    
    # Save predictions to CSV
    df = pd.DataFrame(all_predictions, columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)

    # Plot training metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Vérification des données
def show_sample(image_tensor, label):
    plt.imshow(image_tensor.numpy().transpose(1, 2, 0))
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# Exemple pour la première image
show_sample(X_train_tensor[0], y_train_tensor[0].item())

print("Valeurs des pixels (min, max) :", X_train_tensor.min(), X_train_tensor.max())

# Distribution des classes
unique, counts = np.unique(y_train_tensor.numpy(), return_counts=True)
print("Distribution des classes :", dict(zip(unique, counts)))

# Simplification du modèle pour debugging
class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 10)  # 32x32 images
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = DebugModel().to(device)

# Entraînement sur un sous-ensemble de 100 échantillons
small_train = TensorDataset(X_train_tensor[:100], y_train_tensor[:100])
small_loader = DataLoader(small_train, batch_size=10, shuffle=True)

# Réinitialisation des poids
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

model.apply(reset_weights)

# Désactivation de l'augmentation de données
transform = transforms.Compose([
    ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# Utilisation d'un Learning Rate Finder
# lr_finder = torch_lr_finder.LRFinder(model, optimizer, criterion, device="cuda")
# lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
# lr_finder.plot()

# Vérification matérielle
print("Device utilisé :", device)
print("Mémoire GPU allouée :", th.cuda.memory_allocated() / 1e9, "Go")

# Test avec un jeu de données public (CIFAR-10)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# train_loader = DataLoader(trainset, batch_size=128, shuffle=True)

