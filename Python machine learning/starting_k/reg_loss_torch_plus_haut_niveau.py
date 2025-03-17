import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import torch as th
from tqdm import tqdm
import torch.optim as optim
from torch.nn import functional as F

# Chargement du dataset Iris
iris = datasets.load_iris()

# Selection des deux premiers attributs et remplacement de la classe 2 par la classe 1
X = iris.data[:,:2]
y = iris.target
y[y==2] = 1

# Affichage du dataset
plt.scatter(X[y==0,0],X[y==0,1],color="b")
plt.scatter(X[y==1,0],X[y==1,1],color="r")
plt.show()

N = X.shape[0]
d = X.shape[1]


#Création du modèle de régression logistique. Il étend la classe th.nn.Module de la librairie Pytorch
class Regression_logistique(th.nn.Module):

    # Constructeur qui initialise le modèle
    def __init__(self,d):
        super(Regression_logistique, self).__init__()

        self.layer = th.nn.Linear(d,1)
        self.layer.reset_parameters()

    # Implémentation de la passe forward du modèle
    def forward(self, x):
        out = self.layer(x)
        return F.sigmoid(out)


# Fonction qui calcule les prédictions (0 ou 1) à partir des sorties du modèle
def prediction(f):
    return f.round()

# Fonction qui calcule le taux d'erreur en comparant le y prédit avec le y réel
def error_rate(y_pred,y):
    return ((y_pred != y).sum().float())/y_pred.size()[0]



# Affichage d'une carte des décisions prises par le modèle
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])

def plot_decision(X,y,model,device):

    h = 0.1

    xx1, xx2 = np.meshgrid(np.arange(0,10,h),np.arange(0,10,h))


    xx1_flat = xx1.reshape(xx1.shape[0]**2,1)
    xx2_flat = xx2.reshape(xx2.shape[0]**2,1)

    X_entry = np.hstack((xx1_flat,xx2_flat))

    X_entry = th.from_numpy(X_entry).float().to(device)

    f = model(X_entry)
    y_pred = prediction(f).detach().cpu().numpy()

    yy = y_pred.reshape(xx1.shape[0],xx1.shape[1])

    plt.pcolormesh(xx1,xx2,yy,cmap=cmap_light)

    plt.scatter(X[y==0,0],X[y==0,1],color="r")
    plt.scatter(X[y==1,0],X[y==1,1],color="g")

    plt.show()



#Séparation aléatoire du dataset en ensemble d'apprentissage (70%) et de test (30%)

indices = np.random.permutation(X.shape[0])

training_idx, test_idx = indices[:int(X.shape[0]*0.7)], indices[int(X.shape[0]*0.7):]

X_train = X[training_idx,:]
y_train = y[training_idx]

X_test = X[test_idx,:]
y_test = y[test_idx]



#Spécification du materiel utilisé device = "cpu" pour du calcul CPU, device = "cuda:0" pour du calcul sur le device GPU "cuda:0".
device = "cpu"

# Instanciation du modele de régression logistique et chargement sur le device
model = Regression_logistique(d)
model = model.to(device)

# Conversion des données en tenseurs Pytorch et envoi sur le device
X_train = th.from_numpy(X_train).float().to(device)
y_train = th.from_numpy(y_train).float().to(device).unsqueeze(1)

X_test = th.from_numpy(X_test).float().to(device)
y_test = th.from_numpy(y_test).float().to(device).unsqueeze(1)

#Taux d'apprentissage (learning rate)
eta = 0.001

# ###### 1ère version détaillée
# Apprentissage du modèle et calcul de la performance tous les 100 itérations
nb_epochs = 10000

# Définition du critère de Loss. Ici BCELoss correspond à la fonction de binary cross entropy qu'on avait définie plus haut
criterion = th.nn.BCELoss()

# optim.SGD Correspond à la descente de gradient standard qu'on a vu en cours.
# Il existe d'autres types d'optimizer dans la librairie Pytorch
# Le plus couramment utilisé est optim.Adam
optimizer = optim.SGD(model.parameters(), lr=eta)

# tqdm permet d'avoir une barre de progression
pbar = tqdm(range(nb_epochs))


for i in pbar:
    # Remise à zéro des gradients à chaque epoch
    optimizer.zero_grad()

    f_train = model(X_train)

    #Calcul de la loss
    loss = criterion(f_train,y_train)

    # Calculs des gradients
    loss.backward()

    # Mise à jour des paramètres du modèle suivant l'algorithme d'optimisation retenu
    optimizer.step()

    if(i%500==0):

        y_pred_train = prediction(f_train)
        error_train = error_rate(y_pred_train,y_train)
        loss = criterion(f_train,y_train)

        f_test = model(X_test)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, y_test)
        plot_decision(X, y, model, device)

        pbar.set_postfix(iter=i, loss = loss.item(), error_train=error_train.item(), error_test=error_test.item())





