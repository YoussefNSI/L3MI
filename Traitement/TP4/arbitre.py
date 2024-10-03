import pandas as pd
import string
from pprint import pprint

def lire(filename, compPerequation=[], compClassique=[]):
    df = pd.read_csv(filename)
    listeDomicile = []
    listeVisiteur = []
    listeClub = {} # dict avec comme clé le code club et comme valeur le nom du club
    listeArbitres = []
    dictIndemnites = {}
    listeRencontres = []
    factureClub = {}
    rencontres = df[["lb_nom_abg", "GS1", "GS2", "id", "nom", "prenom", "numero_licence", "libelle", "indemnite", "presence"]]
    for i in range(len(rencontres)):
        rencontre = rencontres.iloc[i]
        Club1 = [part.strip() for part in rencontre['GS1'].split('-')] # strip retire les espaces
        Club2 = [part.strip() for part in rencontre['GS2'].split('-')]
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
                if str(rencontre["indemnite"]).lower() != 'nan':
                    dictIndemnites[rencontre["id"]] = [listeArbitres[-1][4], listeArbitres[-1][3]] # [indemnite, numero_licence]
                    listeRencontres.append(rencontre) # tri des rencontres pour les factures (arbitres présents et indemnisés seulement)
            else:
                for arbitre in listeArbitres:
                    if arbitre[0] == rencontre["id"]:
                        if str(rencontre["indemnite"]).lower() != 'nan':
                            arbitre[4] += rencontre["indemnite"] 
                            dictIndemnites[rencontre["id"]][0] += rencontre["indemnite"]
                            listeRencontres.append(rencontre) # meme chose
                               
    for club in listeClub:
        factureClub[club] = 0

    for match in listeRencontres:
        if match['lb_nom_abg'] in compPerequation:
            continue # on ajoute pas l'indemnité à la facture du club car déjà dans le pot commun si j'ai bien compris
            # perequation
        else:
            factureClub[match['GS1'].split('-')[0].strip()] += round(match['indemnite']/2, 2)
            factureClub[match['GS2'].split('-')[0].strip()] += round(match['indemnite']/2, 2)
            # classique
            continue
        
    return (dictIndemnites, factureClub)
    
lire("officiels.csv")
    