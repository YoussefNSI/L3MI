import pandas as pd
from pprint import pprint

df = pd.read_csv('Annales/fete-de-la-musique-2019.csv', sep=';')

# Filter concerts with age minimum >= 18
# fillna(0) replaces NaN values with 0
# astype(int) converts the "Age minimum" column to integers
# Filter the DataFrame to get rows where "Age minimum" is >= 18
adult_concerts = df[df["Age minimum"].fillna(0).astype(int) >= 18]
for i, concert in adult_concerts.iterrows():
    print(i, " ", concert["Titre (fr)"])

# Count concerts by region
# dropna() removes NaN values from the "Région" column
# value_counts() counts the occurrences of each region
dictRegion = df["Région"].dropna().value_counts().to_dict()

# Count labels
# dropna() removes NaN values from the "label_multivalued" column
# str.split(';') splits the labels into lists
# explode() transforms these lists into separate rows
# value_counts() counts the occurrences of each label
dictLabels = df["label_multivalued"].dropna().str.split(';').explode().value_counts().to_dict()

print(dictLabels.keys())
pprint(dictRegion)
pprint(dictLabels)


import pandas as pd
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv("data.tsv", sep="\t")

# Liste des pays présents dans les données
pays_list = df["pays"].unique()

# Création des figures
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fonction pour tracer un boxplot avec Matplotlib
def boxplot(ax, data, column, title, ylabel):
    grouped_data = [data[data["pays"] == pays][column].dropna() for pays in pays_list]
    ax.boxplot(grouped_data, labels=pays_list, patch_artist=True)
    ax.set_title(title)
    ax.set_xlabel("Pays")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=90)

# Boxplot pour la répartition des âges
boxplot(axes[0], df, "âge", "Répartition des âges par pays", "Âge")

# Boxplot pour la répartition des tailles
boxplot(axes[1], df, "taille", "Répartition des tailles par pays", "Taille (cm)")

# Sauvegarde et affichage du graphique
plt.tight_layout()
plt.savefig("repartition_ages_tailles.png")
plt.show()

