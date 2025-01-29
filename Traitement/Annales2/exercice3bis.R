# Définir la fonction exo3
exo3 <- function(filename) {
  # Charger les données
  data <- read.csv(filename, sep = ";")
  
  # 1. Afficher la répartition des indices
  distribution <- table(data$indice)
  print("Répartition des indices :")
  print(distribution)
  
  # 2. Créer un graphique en camembert
  png("r_repartition_indices.png")  # Ouvrir un fichier PNG pour sauvegarder le graphique
  pie(distribution, main = "Répartition des indices")  # Créer le camembert
  dev.off()  # Fermer le fichier PNG
  
  # 3. Afficher la répartition des indices par communes
  commune_distribution <- table(data$commune_nom, data$indice)
  print("Répartition des indices par communes :")
  print(commune_distribution)
  
  # 4. Trouver la commune avec le moins de "bon" qualité de l'air
  min_bon_commune <- names(which.min(table(data$commune_nom[data$indice == "bon"])))
  print("Commune où il y a le moins de bonne qualité de l'air :")
  print(min_bon_commune)
}

# Appeler la fonction exo3 avec le fichier CSV
exo3("Annales2/dataairplqualiteairangers.csv")