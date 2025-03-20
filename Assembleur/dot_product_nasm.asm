section .text
global main
global dp_fpu
extern printf

; Format string pour l'affichage
section .data
format db "Produit scalaire: %f", 10, 0  ; 10 = nouvelle ligne, 0 = fin de chaîne

; Notre fonction de produit scalaire existante
dp_fpu:
    push ebp
    mov ebp, esp
    push esi
    push edi

    ; Récupération des paramètres
    mov esi, [ebp+8]  ; esi = x (vecteur 1)
    mov edi, [ebp+12] ; edi = y (vecteur 2)
    mov edx, [ebp+16] ; edx = size (nombre d'éléments)

    ; Initialisation de la somme à zéro
    fldz            ; st0 = 0.0f (somme initiale)
    
    ; Initialisation du compteur de boucle
    xor ecx, ecx    ; ecx = 0 (index i)

.for:
    ; Condition de sortie de boucle
    cmp ecx, edx    ; comparer i et size
    jge .endfor     ; si i >= size, sortir de la boucle

    ; Calcul de x[i] * y[i]
    fld dword [esi + ecx*4]  ; st0 = x[i], st1 = sum
    fmul dword [edi + ecx*4] ; st0 = x[i] * y[i], st1 = sum
    
    ; Addition à la somme
    faddp st1, st0            ; st0 = sum + x[i]*y[i]

    ; Incrémentation du compteur
    inc ecx         ; i++
    jmp .for        ; retour au début de la boucle

.endfor:
    ; La somme est déjà dans st0, prête à être retournée
    
    ; Restauration des registres
    pop edi
    pop esi
    
    ; Épilogue standard
    mov esp, ebp
    pop ebp
    ret

; Nouvelle fonction main
main:
    push ebp
    mov ebp, esp
    
    ; Exemple : calculer le produit scalaire de deux vecteurs de test
    ; (il faudrait définir ces vecteurs dans la section .data)
    
    ; Appel à dp_fpu
    push 4          ; taille = 4
    push y          ; pointeur vers y
    push x          ; pointeur vers x
    call dp_fpu
    add esp, 12     ; nettoyer la pile (3 arguments * 4 octets)
    
    ; Afficher le résultat (st0 contient le résultat)
    sub esp, 8      ; réserver de l'espace pour le double sur la pile
    fstp qword [esp] ; transférer le résultat flottant vers la pile
    push format     ; chaîne de format pour printf
    call printf
    add esp, 12     ; nettoyer la pile
    
    ; Retourner 0
    xor eax, eax
    
    mov esp, ebp
    pop ebp
    ret

; Définir des vecteurs de test
section .data
vecteur_x dd 1.0, 2.0, 3.0
vecteur_y dd 4.0, 5.0, 6.0

; Définition des vecteurs de test
x: dd 1.0, 2.0, 3.0, 4.0
y: dd 5.0, 6.0, 7.0, 8.0
format db "%f", 0