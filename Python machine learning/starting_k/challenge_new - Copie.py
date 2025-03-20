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

if test_CNN:
    # Completely remove multiprocessing setup and simplify data loading
    
    # Preprocess data before training to avoid DataLoader transform issues
    print("Preprocessing training data...")
    
    # Convert data to PIL images for transformations
    from PIL import Image
    import io
    
    def tensor_to_pil(tensor):
        """Convert a tensor to PIL Image."""
        return ToPILImage()(tensor)
    
    def apply_transforms(img_tensor, transform):
        """Apply transforms to an image tensor and return a tensor."""
        img_pil = tensor_to_pil(img_tensor)
        return transform(img_pil)
    
    # More balanced data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # Removed vertical flip as it's less relevant for natural images
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # Moderate augmentation
        transforms.RandomRotation(10),  # Moderate rotation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))  # Moderate erasing
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])
    
    # Pre-transform data (this will take some time but avoid worker issues)
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
    
    # Create datasets from pre-transformed data
    train_dataset = TensorDataset(train_imgs, train_labels)
    val_dataset = TensorDataset(val_imgs, val_labels)
    
    # Create data loaders WITHOUT workers (safe for Windows)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,
        shuffle=True,
        num_workers=0,  # Disable multiprocessing completely
        pin_memory=True if th.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=256,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing completely
        pin_memory=True if th.cuda.is_available() else False
    )
    
    # Balanced CNN with moderate capacity and regularization
    class BalancedCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(BalancedCNN, self).__init__()
            
            # First convolutional block - light regularization
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.dropout1 = nn.Dropout2d(0.1)  # Light dropout
            
            # Second block - moderate capacity
            self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn2a = nn.BatchNorm2d(128)
            self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn2b = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.dropout2 = nn.Dropout2d(0.2)  # Moderate dropout
            
            # Third block - slightly higher capacity
            self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn3a = nn.BatchNorm2d(256)
            self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn3b = nn.BatchNorm2d(256)
            self.pool3 = nn.MaxPool2d(2, 2)
            self.dropout3 = nn.Dropout2d(0.25)  # Moderate dropout
            
            # Global pooling and classifier with moderate regularization
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(256, 256)
            self.bn_fc = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, num_classes)
            self.dropout_fc = nn.Dropout(0.3)  # Moderate dropout
            
        def forward(self, x):
            # Block 1
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = self.dropout1(x)
            
            # Block 2 with residual-like connection
            identity = x
            x = F.relu(self.bn2a(self.conv2a(x)))
            x = self.bn2b(self.conv2b(x))
            if hasattr(self, 'downsample2'):
                identity = self.downsample2(identity)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.dropout2(x)
            
            # Block 3 with residual-like connection
            x = F.relu(self.bn3a(self.conv3a(x)))
            x = self.bn3b(self.conv3b(x))
            x = F.relu(x)
            x = self.pool3(x)
            x = self.dropout3(x)
            
            # Global pooling and classification
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.bn_fc(self.fc1(x)))
            x = self.dropout_fc(x)
            x = self.fc2(x)
            
            return x
    
    # Initialize model
    model = BalancedCNN().to(device)
    
    # Moderate weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
    
    # Moderate regularization with Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Moderate weight decay
    
    # OneCycle learning rate scheduler
    from torch.optim.lr_scheduler import OneCycleLR
    num_epochs = 100
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.005,  # Peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.2,  # Spend 20% of time warming up
        div_factor=25.0,  # Initial learning rate = max_lr/div_factor
        final_div_factor=10000.0,  # Final learning rate = max_lr/final_div_factor
    )
    
    # Moderate label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Reduced label smoothing
    
    # Reduced mixup probability and alpha
    mixup_prob = 0.3  # Only 30% of batches use mixup
    
    def mixup_data(x, y, alpha=0.2):  # Lower alpha for less aggressive mixing
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = th.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    # Tracking metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping parameters
    best_val_loss = float('inf')
    best_val_acc = 0.0  # Also track best validation accuracy
    best_model = None
    patience = 15  # Increased patience
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Reduced mixup probability
            use_mixup = np.random.random() < mixup_prob
            if use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Apply loss with mixup if used
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            th.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * images.size(0)
            
            # For accuracy calculation, only consider non-mixup batches
            if not use_mixup:
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Calculate training metrics
        train_loss_epoch = train_loss / len(train_loader.dataset)
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
        
        # Update learning rate
        scheduler.step(val_loss_epoch)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")
        
        # Save best model based on validation accuracy instead of loss
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
            best_model = model.state_dict().copy()
            patience_counter = 0
            print(f"New best model saved! Validation accuracy: {val_acc_epoch:.4f}")
        else:
            patience_counter += 1
            print(f"Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
        
        # Print learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Load the best model for testing
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Load test data
    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X_test = data['data']
        
    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0).to(device)
    
    # Predict
    model.eval()
    with th.no_grad():
        outputs = model(X_test_tensor)
        y_pred = prediction(outputs)
    
    # Save predictions
    df = pd.DataFrame(y_pred.cpu().numpy(), columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)
    
    # Plot the training curves
    plt.figure(figsize=(12, 6))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='green')
    plt.plot(val_accuracies, label='Val Accuracy', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # After model training and loading best weights, add test-time augmentation
    def test_time_augmentation(model, image, n_aug=10):
        """Apply test-time augmentation to improve predictions"""
        model.eval()
        
        # Get base prediction
        with th.no_grad():
            base_pred = model(image.unsqueeze(0))
            probs = F.softmax(base_pred, dim=1)
        
        # Apply horizontal flip
        flipped = th.flip(image, dims=[2])
        with th.no_grad():
            flip_pred = model(flipped.unsqueeze(0))
            probs += F.softmax(flip_pred, dim=1)
        
        # Apply small shifts
        transforms_list = []
        # Shift right
        shifted = F.pad(image[:, :, :-1], (1, 0, 0, 0), mode='replicate')
        transforms_list.append(shifted)
        
        # Shift left
        shifted = F.pad(image[:, :, 1:], (0, 1, 0, 0), mode='replicate')
        transforms_list.append(shifted)
        
        # Shift down
        shifted = F.pad(image[:, :-1, :], (0, 0, 1, 0), mode='replicate')
        transforms_list.append(shifted)
        
        # Shift up
        shifted = F.pad(image[:, 1:, :], (0, 0, 0, 1), mode='replicate')
        transforms_list.append(shifted)
        
        # Apply minor brightness variations
        brighten = image * 1.1
        brighten = th.clamp(brighten, 0, 1)
        transforms_list.append(brighten)
        
        darken = image * 0.9
        transforms_list.append(darken)
        
        # Process all transformations
        for aug in transforms_list:
            with th.no_grad():
                aug_pred = model(aug.unsqueeze(0))
                probs += F.softmax(aug_pred, dim=1)
        
        # Average predictions (1 original + 1 flip + 6 transforms)
        probs /= (8)
        return probs.argmax(dim=1).item()
    
    # Replace the test prediction code with TTA
    # Load test data
    with open("data_images_test", 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    X_test = data['data']
        
    X_test_tensor = th.tensor(X_test.reshape(-1, 3, 32, 32), dtype=th.float32).div_(255.0)
    
    # Normalize test data
    X_test_normalized = []
    for img in X_test_tensor:
        normalized = transforms.Normalize(mean=mean.tolist(), std=std.tolist())(img)
        X_test_normalized.append(normalized)
    X_test_normalized = th.stack(X_test_normalized).to(device)
    
    # Apply test-time augmentation to each test sample
    model.eval()
    y_pred = []
    for i in tqdm(range(len(X_test_normalized)), desc="Predicting with test-time augmentation"):
        # Batch prediction for efficiency
        if i % 100 == 0:
            print(f"Processing test images {i}-{min(i+100, len(X_test_normalized))} of {len(X_test_normalized)}")
            
        # Apply TTA to each image
        pred = test_time_augmentation(model, X_test_normalized[i])
        y_pred.append(pred)
    
    # Save predictions
    df = pd.DataFrame(y_pred, columns=["target"])
    df["target"] = df["target"].apply(lambda x: f"{x:.18e}")
    df.to_csv("images_test_predictions.csv", index=False, header=False)
    
    # Plot final training curves
    plt.figure(figsize=(15, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot training and validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    # Plot learning rate
    lrs = []
    for i in range(len(train_losses)):
        optimizer.param_groups[0]['lr'] = 0.001 * (0.9 ** i)  # Approximate LR schedule
        lrs.append(optimizer.param_groups[0]['lr'])
    
    plt.subplot(1, 3, 3)
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

