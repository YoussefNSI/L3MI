import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F

#0) Choix d'une graine aléatoire
np.random.seed(0)
th.manual_seed(0)
th.cuda.manual_seed(0)





#1) Chargement des données
iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

d = X.shape[1]
k = 3

#2) fonction qui calcule les prédictions (0 ou 1) à partir des sorties du modèle
def prediction(f):
    return th.argmax(f, 1)

#3) Fonction qui calcule le taux d'erreur en comparant les y prédits avec les y réels
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]


#4) Affichage des frontières de décision
def plot_decision(X,y,model):
    h = 0.05

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    x_flat = xx.ravel()
    y_flat = yy.ravel()

    x_flat = x_flat.reshape(x_flat.shape[0],1)
    y_flat = y_flat.reshape(y_flat.shape[0], 1)

    X_entry = np.hstack((x_flat,y_flat))

    X_entry = th.from_numpy(X_entry).float().to(device)

    f = model(X_entry)
    y_pred = prediction(f).detach().cpu().numpy()
    preds = y_pred.reshape(xx.shape)


    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure()
    plt.pcolormesh(xx, yy, preds, cmap=cmap_light)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.show()



#5) Séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)
indices = np.random.permutation(X.shape[0])
training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]
X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]





#6) Création du modèle de régression logistique multivarié. Il étend la classe th.nn.Module de la librairie Pytorch
class Reg_log_multi(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,d,k):
        super(Reg_log_multi, self).__init__()

        self.layer = th.nn.Linear(d,k)
        self.layer.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        out = self.layer(x)
        return F.softmax(out,1)


#7) creation d'un objet modele de régression logistique multivarié avec les paramètres d et k
model = Reg_log_multi(d,k)

#8) Spécification du materiel utilisé device = "cpu" pour du calcul CPU, device = "cuda:0" pour du calcul sur le device GPU "cuda:0".
device = "cpu"

#9) Passage du modèle sur le device
model = model.to(device)


#10) Conversion des données en tenseurs Pytorch et envoi sur le device
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).long().to(device)

X_test = th.from_numpy(X_test).float().to(device)
y_test = th.from_numpy(y_test).long().to(device)


#11) Taux d'apprentissage (learning rate)
eta = 0.01

#12) Définition du critère de Loss. Ici cross entropy pour un modèle de classification multi classe
criterion = th.nn.CrossEntropyLoss()

# optim.SGD Correspond à la descente de gradient standard.
# Il existe d'autres types d'optimizer dans la librairie Pytorch
# Le plus couramment utilisé est optim.Adam
optimizer = optim.SGD(model.parameters(), lr=eta)

# tqdm permet d'avoir une barre de progression
nb_epochs = 100000
pbar = tqdm(range(nb_epochs))

for i in pbar:
    # Remise à zéro des gradients
    optimizer.zero_grad()

    f_train = model(X_train)
    loss = criterion(f_train,y_train)
    # Calculs des gradients
    loss.backward()

    # Mise à jour des poids du modèle avec l'optimiseur choisi et en fonction des gradients calculés
    optimizer.step()

    if (i % 1000 == 0):

        y_pred_train = prediction(f_train)

        error_train = error_rate(y_pred_train,y_train)
        loss = criterion(f_train,y_train)

        f_test = model(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())

    if (i % 5000 == 0):
        plot_decision(X, y, model)





