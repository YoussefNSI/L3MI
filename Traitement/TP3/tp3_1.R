library(dplyr)

officiels <- read.csv("officiels.csv")

clubs <- grep("-*-", unique(officiels$GS1), ignore.case = TRUE, value = TRUE)
print(colnames(officiels))

