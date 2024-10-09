concerts <- read.csv("fete-de-la-musique-2019.csv", sep = ";", 
                     stringsAsFactors = FALSE)

concerts <- concerts[, c("Titre..fr.", "Age.minimum", "Région", 
                         "label_multivalued")]

dict_region <- c()
dict_labels <- c()

for (i in seq_len(nrow(concerts))) {
  concert <- concerts[i, ]
  
  if (!is.na(as.numeric(concert["Age.minimum"]))) {
    if (as.numeric(concert["Age.minimum"]) >= 18) {
      print(paste(i, concert["Titre..fr."]))
    }
  }
  
  if (!is.na(concert[["Région"]]) && concert[["Région"]] != "NA") {
    region <- concert[["Région"]]
    if (!(region %in% names(dict_region))) {
      dict_region[region] <- 0
    }
    dict_region[region] <- dict_region[region] + 1
  }
  
  if (!is.na(concert[["label_multivalued"]])) {
    labels <- unlist(strsplit(concert[["label_multivalued"]], ";"))
    for (label in labels) {
      if (!(label %in% names(dict_labels))) {
        dict_labels[label] <- 0
      }
      dict_labels[label] <- dict_labels[label] + 1
    }
  }
}

dict_region <- Filter(function(x) x != "", dict_region)
dict_region <- dict_region[names(dict_region) != ""]


print(names(dict_labels))
print(dict_region)
print(dict_labels)
