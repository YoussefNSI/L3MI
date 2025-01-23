**Fiche de révision : Assembleur x86**

---

### 1. **Notions de base**
- **Registres principaux** :
  - `EAX`, `EBX`, `ECX`, `EDX` : Registres généraux.
  - `ESI`, `EDI` : Utilisés pour les opérations de chaîne (source et destination).
  - `EBP` : Base pointer (pointeur de pile pour accéder aux variables locales).
  - `ESP` : Stack pointer (pointeur de la pile).
  - `EIP` : Instruction pointer (adresse de la prochaine instruction).

- **Registres segmentaires** :
  - `CS`, `DS`, `ES`, `SS`, `FS`, `GS` : Utilisés pour segmenter la mémoire.

- **Taille des opérandes** :
  - `BYTE` : 8 bits (1 octet).
  - `WORD` : 16 bits (2 octets).
  - `DWORD` : 32 bits (4 octets).

### 2. **Instructions essentielles**

#### a) Transfert de données
- `MOV dest, src` : Copie la valeur de `src` vers `dest`.
- `PUSH val` : Empile `val` sur la pile.
- `POP dest` : Dépile une valeur de la pile dans `dest`.
- `LEA dest, [src]` : Charge l'adresse effective de `src` dans `dest`.

#### b) Opérations arithmétiques
- `ADD dest, src` : Additionne `src` à `dest`.
- `SUB dest, src` : Soustrait `src` de `dest`.
- `MUL src` : Multiplie `EAX` par `src`, résultat dans `EDX:EAX`.
- `DIV src` : Divise `EDX:EAX` par `src`, quotient dans `EAX`, reste dans `EDX`.
- `INC dest` : Incrémente `dest` de 1.
- `DEC dest` : Décrémente `dest` de 1.

#### c) Opérations logiques
- `AND dest, src` : Opération ET logique.
- `OR dest, src` : Opération OU logique.
- `XOR dest, src` : Opération XOR logique.
- `NOT dest` : Complément logique.
- `SHL dest, count` : Décalage à gauche (multiplication par 2^count).
- `SHR dest, count` : Décalage à droite (division par 2^count).

#### d) Comparaison et saut
- `CMP op1, op2` : Compare `op1` à `op2` (résultat dans les flags).
- `JE addr` : Saut si égal (ZF = 1).
- `JNE addr` : Saut si différent (ZF = 0).
- `JG addr` : Saut si supérieur (ZF = 0 et SF = OF).
- `JL addr` : Saut si inférieur (SF != OF).
- `JMP addr` : Saut inconditionnel.

#### e) Appels de fonctions
- `CALL addr` : Appelle une fonction (empile l'adresse de retour).
- `RET` : Retourne à l'appelant (dépile l'adresse de retour).

### 3. **Gestion de la pile**
- La pile fonctionne en **LIFO** (Last In, First Out).
- La pile est utilisée pour :
  - Sauvegarder des registres (par exemple, avant un appel de fonction).
  - Passer des arguments aux fonctions.
  - Stocker les adresses de retour d'un appel de fonction.
  - Allouer de l'espace pour des variables locales.

**Opérations principales sur la pile :**
- **Empilement (`PUSH`)** :
  - Décrémente `ESP` (le pointeur de pile) de la taille de l'opérande (généralement 4 octets pour `DWORD`) et copie la valeur à l'adresse pointée par `ESP`.

  ```asm
  push eax       ; Empile la valeur de EAX sur la pile
  ```

- **Dépilement (`POP`)** :
  - Copie la valeur à l'adresse pointée par `ESP` dans la destination, puis incrémente `ESP` de la taille de l'opérande.

  ```asm
  pop ebx        ; Dépile la valeur au sommet de la pile dans EBX
  ```

**Structure typique d'une fonction :**
1. **Prologue de fonction** :
   - Sauvegarde le contexte en empilant `EBP` et en configurant une base pour accéder à la pile.
   - Alloue de l'espace pour les variables locales.

   ```asm
   push ebp            ; Sauvegarde l'ancien EBP
   mov ebp, esp        ; EBP pointe sur la base actuelle de la pile
   sub esp, taille     ; Réserve de l'espace pour les variables locales
   ```

2. **Corps de la fonction** :
   - Accès aux variables locales via des offsets de `EBP` (par exemple, `DWORD PTR [EBP-4]`).
   - Arguments passés via des offsets de `EBP` (par exemple, `DWORD PTR [EBP+8]` pour le premier argument).

3. **Épilogue de fonction** :
   - Restaure le contexte et retourne à l'appelant.

   ```asm
   mov esp, ebp        ; Restaure ESP à son état initial
   pop ebp             ; Restaure l'ancien EBP
   ret                 ; Retourne à l'appelant
   ```

**Exemple complet : Fonction qui additionne deux nombres :**
```asm
add_numbers:
    push ebp            ; Sauvegarde l'ancien cadre de pile
    mov ebp, esp        ; Établit un nouveau cadre de pile
    mov eax, [ebp+8]    ; Charge le premier argument (a) dans EAX
    add eax, [ebp+12]   ; Ajoute le deuxième argument (b)
    pop ebp             ; Restaure l'ancien cadre de pile
    ret                 ; Retourne à l'appelant
```

**Passage d'arguments (exemple en C++) :**
```cpp
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 10);
    return result;
}
```
Traduction assembleur :
```asm
section .text
    global _start

_start:
    push 10             ; Empile le second argument
    push 5              ; Empile le premier argument
    call add_numbers    ; Appelle la fonction
    add esp, 8          ; Nettoie la pile (2 arguments * 4 octets)

    ; Stocke le résultat et termine
    mov eax, 1          ; Appel système pour quitter
    int 0x80
```


### 4. **Flags principaux**
- **ZF (Zero Flag)** : Mis à 1 si le résultat est nul.
- **CF (Carry Flag)** : Mis à 1 si une retenue ou un emprunt a lieu.
- **OF (Overflow Flag)** : Indique un dépassement lors d'op