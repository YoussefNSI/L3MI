/*
Fonction renvoyant un tableau d'objets correspondant au contenu d'un fichier CSV
donné sous la forme d'une chaîne de caractères `données`.
`données` contient une ligne d'en-têtes définissant les noms des champs (colonnes) dont
- le séparateur de lignes est `séparateur_lignes`
- le séparateur de colonnes est `séparateur_colonnes`.

Chaque objet du tableau représente une ligne de `données` (à l'exception de la ligne d'en-têtes)
et chacune de ses propriétés correspond au nom d'un champ et à sa valeur dans la ligne.
*/
export async function parseCSV(données, séparateur_lignes, séparateur_colonnes) {
    const csv = [];
    const data = await p_fetch(données, "csv");
    const lignes = data.split(séparateur_lignes);
    const en_têtes = lignes.shift().split(séparateur_colonnes);
    lignes.forEach(ligne => {
        const obj = {};
        const valeurs = ligne.split(séparateur_colonnes);
        en_têtes.forEach((en_tête, i) => {
            obj[en_tête] = valeurs[i];
        });
        csv.push(obj);
    });
    return csv;
};


/* 
Fonction renvoyant une promesse qui extrait par Fetch le contenu d'un fichier 
accessible à l'URL `url` et de format `format`.
`format` est l'une des chaînes :
- "xml" (fichier XML), 
- "json" (fichier JSON)
- "csv" (fichier CSV). 
*/
export const p_fetch = (url, format) =>
    fetch(url)
    .then(réponse => {
        if (!réponse.ok) {
            throw Error(`HTTP error, status = ${réponse.status}`);
        }

        switch (format) {
            case "xml":
            case "csv":
                return réponse.text();
                break;
            case "json":
                return réponse.json();
                break;
            default:
                throw Error(`Format de fichier incorrect = ${format}`);

        }

    })
    .catch(erreur => {
        console.log(erreur.message);
    });