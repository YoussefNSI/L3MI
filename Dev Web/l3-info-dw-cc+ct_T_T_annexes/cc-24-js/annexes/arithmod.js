import { rgb, générerCasesCongruence } from "./utils.js";
import { ZnZ } from "./ZnZ.js";
import * as solution from "./solution.js";


/////////////////////// MAIN ////////////////////////////////////////////////

// NE PAS MODIFIER CETTE SECTION

let cord = document.querySelector(".c-ordre");
let ccal = document.querySelector(".c-calcul");
let cinv = document.querySelector(".c-inversibles");
let cchi = document.querySelector(".c-chinois");
let crsa = document.querySelector(".c-rsa");

main();

async function main() {
    // ORDRE
    await générerTableau1(2);   // génère le premier tableau HTML

    // CALCUL
    bornerChampsCalcul();       // borne les champs de calcul
    actualiserChampsCalcul();   // actualise les champs de calcul

    // INVERSIBLES
    await générerTableau2();    // génère le second tableau HTML
    écouterTableau2();          // écoute le second tableau HTML

    écouterN();                 // écoute le champ "n"

    // CHINOIS
    await générerN1N2();        // génère les menus déroulants "n1" "n2"
    écouterN1N2();              // écoute les menus "n1" et "n2"
    await générerTableau3();    // génère le troisième tableau HTML
    écouterTableau3();          // écoute le troisième tableau HTML

    // RSA
    await générerNphiN();       // génère les champs "n" et "phi(n)"
    écouterE();                 // écoute le champ "e"
    écouterM();                 // écoute le champ "M"

    // CUSTOMISER
    commuter();                 // commute les conteneurs

    return true;
}

/////////////////////// ORDRE ////////////////////////////////////////////////

async function générerTableau1(k) {
    // Q1.1
    // Crée ou remet à jour l'unique ligne TR du premier tableau HTML 
    // pour afficher les `k` classes de congruence de l'anneau Z/kZ.
    // - S'appuie sur l'utilitaire `générerCasesCongruence()` pour générer et classer les cases.
    // - Ne supprime jamais le tableau !
    solution.générerTableau1(k);
}

function écouterN() {
    // NE PAS MODIFIER
    // A chaque fois que l'utilisateur modifie l'ordre `n`, regénére les tableaux HTML 
    // 1 et 2 et écoute le second par appel aux fonctions `générerTableau1`, `générerTableau2` et
    // `écouterTableau2`.
    solution.écouterN();
}


/////////////////////// CALCUL ////////////////////////////////////////////////

function bornerChampsCalcul() {
    // NE PAS MODIFIER
    // Borne à `n-1` tous les champs de classe "argument" et "résultat" à chaque fois que
    // l'utilisateur modifie l'ordre `n`.
    solution.bornerChampsCalcul();
}

function actualiserChampsCalcul() {
    // Q1.2
    // Toutes les 3 secondes :
    // 1) Génère et affiche des valeurs pour les 4 champs "argument" par tirage aléatoire dans 0..`n`-1.
    // 2) Calcule le résultat des 2 opérations + et * sur ces valeurs en utilisant les méthodes 
    // `plus` et `times` de la classe `ZnZ` et les affiche dans les 2 champs "résultat".
    // 3) Applique la couleur de fond renvoyée par l'utilitaire `rgb` aux 6 champs.
    solution.actualiserChampsCalcul();
}


/////////////////////// INVERSIBLES ////////////////////////////////////////////////

async function générerTableau2() {
    // Q2.1
    // Crée ou remet à jour le second tableau HTML qui est identique au premier
    // mais a pour classes CSS "c-inversibles-1" et "math".    solution.générerTableau2();
    solution.générerTableau2();
}

function écouterTableau2() {
    // Q2.2
    // A chaque clic sur une des cases internes du second tableau HTML :
    // 1. Requête le script pgcd.php en HTTP POST en lui communiquant les clés "x" et "y" 
    // fixées respectivement à 
    // - l'ordre `n` choisi par l'utilisateur
    // - et à l'entier `v` correspondant à la classe de congruence affichée dans la case cliquée.
    // Le script renvoie un objet JSON à propriétés entières de clés "x", "y", "gcd", "cx" et "cy" 
    // où `gcd = PGCD(x,y) = cx * x + cy * y`. 
    // 2. En cas de succès, extrait la réponse JSON puis 
    // - écrase le contenu du paragraphe qui suit le tableau avec une phrase indiquant si `v` est 
    // inversible modulo `n` (quand le PGCD de `v` et `n` vaut 1) ou non
    // - colorie le fond de la case en vert ou en rouge selon l'inversibilité obtenue

    solution.écouterTableau2();
}


/////////////////////// RESTES CHINOIS ////////////////////////////////////////////////

async function générerN1N2() {
    // Q3.1
    // Requête en HTTP GET le fichier primes.json qui contient les 100 premiers entiers premiers.
    // En cas de succès, génère les options des 2 menus déroulants "n1" et "n2" : 
    // - ne conserve que les 30 premiers entiers de la liste
    // - pré-sélectionne le premier entier (resp. second) pour le premier menu (resp. second)
    return solution.générerN1N2();
}

function écouterN1N2() {
    // NE PAS MODIFIER
    // Remet à jour du troisième tableau HTML à chaque modification des champs "n1" ou "n2".
    solution.écouterN1N2();
}

async function générerTableau3() {
    // NE PAS MODIFIER
    // Génère ou remet à jour le troisième tableau HTML (matrice `n1xn2` avec en-têtes)
    // à partir des valeurs renseignés pour les champs "n1" et "n2"
    // - s'appuie sur l'utilitaire `générerMatrice()` pour générer les cases d'en-têtes.
    // - assigne la chaîne "(i,j)" à l'attribut "title" de chaque case (`i`,`j`).
    solution.générerTableau3();
}

function écouterTableau3() {
    // Q3.2
    // Traite tout clic sur le troisième tableau HTML comme suit :
    // - ignore le clic s'il porte sur 1 case d'en-tête, sinon
    // - extrait les valeurs `n1` et `n2` des champs correspondants
    // - extrait les numéros de ligne `a1` et de colonne `a2` de la case ciblée.
    // Ces numéros sont enregistrés dans son attribut "title" au format a1:a2.
    // - appelle la méthode `chineseTheorem()` de la classe `ZnZ` (sans fournir l'arguement optionnel) 
    // - renseigne les champs "n", "m1", "m2", "a1" et "a2" en utilisant l'objet renvoyé
    // - écrase le contenu texte de la case en appelant la méthode `modulo` de `ZnZ`
    // avec la valeur de la propriété "x" de l'objet réponse.
    solution.écouterTableau3();
}



/////////////////////// RSA ////////////////////////////////////////////////

async function générerNphiN() {
    // NE PAS MODIFIER
    // Complète les champs "n" et "phi(n)" à partir des valeurs assignées aux champs "p" et "q".
    solution.générerNphiN();
}

function écouterE() {
    // Q4.1
    // A chaque fois que l'utilisateur modifie l'exposant `e` :
    // 1. vérifie qu'il est premier avec `phi(n)` en utilisant la méthode statique `pgcd` de la classe `ZnZ`
    // (test si PGCD(e,phi(n))==1) sinon l'incrémente automatiquement et force le champ au premier exposant
    // correct trouvé.
    // 2. extrait l'inverse `d` de `e` modulo `phi(n)` fourni par la méthode `pgcd` 
    // (PGCDE(e,phi(n)) = 1 = d*e + k*phi(n)) et remplit le champ correspondant avec `d` si `d` est positif
    // ou `phi(n) + d` sinon.
    solution.écouterE();
}

function écouterM() {
    // NE PAS MODIFIER
    // A chaque saisie du champ "M" :
    // 1. calcule le chiffrement `C` de `M` en appelant la méthode `exponentiation` de la classe `ZnZ`
    // et complète le champ correspondant.
    // 2. Calcule le déchiffrement de `C` en appelant la méthode "chiffrement" et complète le champ correspondant.
    solution.écouterM();
}


/////////////////////// CUSTOMISER ////////////////////////////////////////////////

function commuter() {
    // Q5.1
    // Au clic sur le bouton "up", déplace le conteneur du haut de page en bas de page 
    // en faisant remonter les autres d'une ligne sur la grille CSS.
    // Utilise la méthode globale getComputedStyle pour accéder en lecture aux propriétés
    // CSS calculées
    solution.commuter();
}