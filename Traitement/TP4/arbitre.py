import pandas as pd
import string
import pprint

def lire(filename, compPerequation="", compClassique=""):
    df = pd.read_csv(filename)
    listeDomicile = []
    listeVisiteur = []
    listeClub = {}
    listeArbitres = []
    factureClub = []
    rencontres = df[["lb_nom_abg", "GS1", "GS2", "id", "nom", "prenom", "numero_licence", "libelle", "indemnite", "presence"]]
    for i in range(len(rencontres)):
        rencontre = rencontres.iloc[i]
        Club1 = rencontre['GS1'].split('-')
        Club2 = rencontre['GS2'].split('-')
        listeDomicile.append(Club1)
        listeVisiteur.append(Club2)
        if len(listeDomicile[i]) == 3:
            if listeDomicile[i][1] not in listeClub:
                listeClub[listeDomicile[i][0]] = listeDomicile[i][1]
            elif listeVisiteur[i][1] not in listeClub:
                listeClub[listeVisiteur[i][0]] = listeVisiteur[i][1]
        else:
            if listeDomicile[i][2] not in listeClub:
                listeClub[listeDomicile[i][0]] = listeDomicile[i][1] + listeDomicile[i][2]
            elif listeVisiteur[i][2] not in listeClub:
                listeClub[listeVisiteur[i][0]] = listeVisiteur[i][1] + listeVisiteur[i][2]
        if rencontre["libelle"] == "Arbitre" and rencontre["presence"] == "P":
            if rencontre["id"] not in listeArbitres:
                listeArbitres.append([rencontre["id"], rencontre["nom"], rencontre["prenom"], rencontre["numero_licence"], rencontre["indemnite"]])
            else:
                for arbitre in listeArbitres:
                    if arbitre[0] == rencontre["id"]:
                        arbitre[4] += rencontre["indemnite"]
    print(listeArbitres)
        
    
lire("officiels.csv")
    