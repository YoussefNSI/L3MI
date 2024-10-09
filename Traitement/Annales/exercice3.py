import pandas as pd
from pprint import pprint

df = pd.read_csv('fete-de-la-musique-2019.csv', sep=';')

concerts = df[["Titre (fr)", "Age minimum", "Région", "label_multivalued"]]

dictRegion = {}
dictLabels = {}

for i in range(len(concerts)):
    concert = concerts.iloc[i]
    if float(concert["Age minimum"]) == concert["Age minimum"]:
        if int(concert['Age minimum']) >= 18:
            print(i," ",concert["Titre (fr)"])
            
    if type(concert["Région"]) != float:
        if concert["Région"] not in dictRegion:
            dictRegion[concert["Région"]] = 0
            
        dictRegion[concert["Région"]] += 1
    
    if type(concert["label_multivalued"]) != float:
        listeLabel = concert["label_multivalued"].split(';')
        for label in listeLabel:
            if label not in dictLabels:
                dictLabels[label] = 0
            dictLabels[label] += 1

print(dictLabels.keys())
pprint(dictRegion)
pprint(dictLabels)
    