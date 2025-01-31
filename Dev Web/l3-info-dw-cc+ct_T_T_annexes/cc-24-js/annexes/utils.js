import * as solution from "./solution.js";


// Renvoie un code couleur aléatoire au format RGB
export function rgb() {
    return solution.rgb();
}

/*
Crée m objets représentant des cases TD où
- chaque case est placée dans la classe HTML "congruence"
- la i+1-ème case (i=0..m-1) a pour contenu la valeur i
et les renvoie dans un tableau JS.
*/
export function générerCasesCongruence(m) {
    return solution.générerCasesCongruence(m);
}
