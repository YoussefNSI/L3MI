import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

X = train_df.iloc[:, :-1]
y = train_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Question 5

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

# Question 6

best_k = 1
best_score = 0

for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Meilleur k : {best_k} (score = {best_score:.2f})")

# Question 7

X_test_entries = pd.read_csv("digits_entries_test.csv").values  # Convertir en numpy array
y_test_true = pd.read_csv("digits_target_test.csv").values.ravel()

# Prédictions avec le meilleur k
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)  # X_train est déjà un numpy array
y_test_pred = knn_best.predict(X_test_entries)

# Performance
accuracy = accuracy_score(y_test_true, y_test_pred)
print(f"Précision sur le test : {accuracy:.2f}")