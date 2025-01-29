import pandas as pd
import matplotlib.pyplot as plt

def exo3(filename):
    # Charger les données
    data = pd.read_csv(filename, delimiter=';')
    
    # Afficher la répartition des indices
    distribution = data['indice'].value_counts()
    print("Répartition des indices :")
    print(distribution)
    
    # Créer un graphique en camembert
    distribution.plot(kind='pie', autopct='%1.1f%%')
    plt.title("Répartition des indices")
    plt.savefig("python_repartition_indices.png")
    plt.close()
    
    # Afficher la répartition des indices par communes
    commune_distribution = data.groupby(['commune_nom', 'indice']).size().unstack()
    print("Répartition des indices par communes :")
    print(commune_distribution)
    
    # Trouver la commune avec le moins de "bon" qualité de l'air
    min_bon_commune = data[data['indice'] == 'bon']['commune_nom'].value_counts().idxmin()
    print("Commune où il y a le moins de bonne qualité de l'air :")
    print(min_bon_commune)

if __name__ == "__main__":
    exo3("Annales2/dataairplqualiteairangers.csv")