import numpy as np

import pickle
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split


with open("dataset_images_train", 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')



X_train, X_valid, y_train, y_valid = train_test_split(
    dict["data"], dict["target"], test_size=0.2,random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()

X_valid = torch.from_numpy(X_valid).float()
y_valid = torch.from_numpy(y_valid).long()


class myDataset(torch.utils.data.Dataset):

    def __init__(self, data, label):
        # Initialise dataset from source dataset
        self.data = data
        self.label  = label

    def __len__(self):
        # Return length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Return one element of the dataset according to its index
        return self.data[idx], self.label[idx]


dataset_train = myDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

nb_epoch = 10000

for epoch in range(nb_epoch):

    for data, target in trainloader:

        print(data.size())
        print(target.size())

