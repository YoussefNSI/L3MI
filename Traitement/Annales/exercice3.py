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
