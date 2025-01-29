# Read the CSV file into a dataframe
concerts <- read.csv("fete-de-la-musique-2019.csv", sep = ";", 
                     stringsAsFactors = FALSE)

# Select relevant columns
concerts <- concerts[, c("Titre..fr.", "Age.minimum", "Région", 
                         "label_multivalued")]

# Initialize dictionaries to count regions and labels
dict_region <- list()
dict_labels <- list()

# Iterate over each concert
for (i in seq_len(nrow(concerts))) {
  concert <- concerts[i, ]
  
  # Check if the concert is for adults (age >= 18)
  age_minimum <- as.numeric(concert["Age.minimum"])
  if (!is.na(age_minimum) && age_minimum >= 18) {
    print(paste(i, concert["Titre..fr."]))
  }
  
  # Count the number of concerts per region
  region <- concert[["Région"]]
  if (!is.na(region) && region != "NA") {
    if (!(region %in% names(dict_region))) {
      dict_region[[region]] <- 0
    }
    dict_region[[region]] <- dict_region[[region]] + 1
  }
  
  # Count the number of concerts per label
  labels <- concert[["label_multivalued"]]
  if (!is.na(labels)) {
    labels <- unlist(strsplit(labels, ";"))
    for (label in labels) {
      if (!(label %in% names(dict_labels))) {
        dict_labels[[label]] <- 0
      }
      dict_labels[[label]] <- dict_labels[[label]] + 1
    }
  }
}

# Filter out empty regions
dict_region <- Filter(function(x) x != "", dict_region)
dict_region <- dict_region[names(dict_region) != ""]

# Print the results
print(names(dict_labels))
print(dict_region)
print(dict_labels)



# Chargement des bibliothèques
library(ggplot2)
library(readr)

# Lecture du fichier TSV
df <- read_tsv("data.tsv")

# Création des graphiques
p1 <- ggplot(df, aes(x = pays, y = âge)) +
  geom_boxplot() +
  labs(title = "Répartition des âges par pays", x = "Pays", y = "Âge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

p2 <- ggplot(df, aes(x = pays, y = taille)) +
  geom_boxplot() +
  labs(title = "Répartition des tailles par pays", x = "Pays", y = "Taille (cm)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Sauvegarde en image
ggsave("repartition_ages_tailles.png", arrangeGrob(p1, p2, ncol = 2), width = 14, height = 6)

