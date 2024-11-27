Pieces1 = [1,2,5,10,20,50,100,200]

# question 1 :

def Plus_Grande_Piece(montant, Pieces):
    plusGrandePiece = 0
    for piece in Pieces:
            if piece <= montant:
                plusGrandePiece = piece
    return plusGrandePiece

#question 2 :

def Decomposition(montant, Pieces):
    resultat = []
    while montant > 0:
        piece = Plus_Grande_Piece(montant, Pieces)
        resultat.append(piece)
        montant -= piece
    return resultat

# Question 3 :

Pieces2 = [1,2,4,5,10,20,50,100,200]
print(Decomposition(8, Pieces2)) # ça retourne [5,2,1], c'est pas la decomposition la plus minimale. Ca devrait être [4,4]

# Question 4 : 


def dec_opt_rec(M, Pieces):
    if M == 0:
        return []
    if M in Pieces:
        return [M]

    dec_min = None
    for piece in Pieces:
        if piece <= M:
            dec_actuelle = dec_opt_rec(M - piece, Pieces)
            
            dec_actuelle = [piece] + dec_actuelle
            
            if dec_min is None or len(dec_actuelle) < len(dec_min):
                dec_min = dec_actuelle
    
    return dec_min

print(dec_opt_rec(8, Pieces2))

# Question 5 :

# Avec 499, le programme ne s'arrête pas. Il faut donc utiliser une méthode itérative pour résoudre le problème.

# Question 6 :

def dec_opt_dyn(M, Pieces):
    dec_min = [None] * (M + 1)
    dec_min[0] = []
    
    for i in range(1, M + 1):
        for piece in Pieces:
            if piece <= i:
                dec_actuelle = dec_min[i - piece]
                if dec_actuelle is not None:
                    dec_actuelle = [piece] + dec_actuelle
                    if dec_min[i] is None or len(dec_actuelle) < len(dec_min[i]):
                        dec_min[i] = dec_actuelle
    
    return dec_min[M]

decompo1 = []
decompo2 = []
for i in range(1,500):
    decompo1.append(dec_opt_dyn(i, Pieces1))
    decompo2.append(dec_opt_dyn(i, Pieces2))


gainTotal = 0
for i in range(0,499):
    gainTotal += len(decompo1[i]) - len(decompo2[i])

print("Le gain total en nombre de pièece grâce a l'ajout de la piece de 4 cts est : ", gainTotal)

def trouver_nouvelle_piece(Pieces):
    piece_a_ajouter = None
    meilleurGain = -1
    
    for nouvellePiece in range(1, 499 + 1):
        if nouvellePiece in Pieces:
            continue
        
        nouvellePieces = Pieces + [nouvellePiece]
        decompo1 = []
        decompo2 = []
        
        for i in range(1, 499 + 1):
            decompo1.append(dec_opt_dyn(i, Pieces))
            decompo2.append(dec_opt_dyn(i, nouvellePieces))
        
        gain_total = 0
        for i in range(499):
            gain_total += len(decompo1[i]) - len(decompo2[i])
        
        if gain_total > meilleurGain:
            meilleurGain = gain_total
            piece_a_ajouter = nouvellePiece
    
    return piece_a_ajouter, meilleurGain

piece, gain = trouver_nouvelle_piece(Pieces1)
print("La piece à ajouter est : ", piece, " et le gain total est : ", gain)


        
    
    







