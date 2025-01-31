let puzzle = null;

fetch('puzzle.json', {
        method: 'GET'
    })
    .then((response) => response.json())
    .then((puzzles) => {
        // Extraction du premier puzzle et construction du générateur
        console.log('Success:', puzzles);
        puzzle = puzzles[0];
        let generator = new PuzzleGenerator(puzzle);

        // Q1.1 Injection de l'auteur
        generator.insertAuthor();

        // Q1.2 Injection des images
        generator.insertImages();

        // Q1.3 Injection de l'énoncé
        generator.insertStatement();

        // Injection des en-têtes du tableau (ne pas faire cette question)
        generator.insertTableHeaders();

        // Q1.4 Injection des indices
        generator.insertHints();

        // Q1.5 Injection des menus déroulants
        generator.insertDropDowns();

        // Q2.1 Clic sur cellules
        generator.handleClicks();

        // Q2.2 Cochage des indices
        generator.handleHints();

        // Q3 Gestion du formulaire
        generator.handleDropDowns();

        return puzzle;
    })
    .catch((error) => {
        console.error('Error:', error);
    });



// Q4 Minuteur
let minuteur = document.querySelector("#minuteur");

function minuter() {
    let temps_passé = new Date() - date_départ;
    if (temps_passé <= temps_imparti) {
        minuteur.innerHTML = "Encore " + Math.floor((temps_imparti - temps_passé) / 1000) + "s"
        let t = setTimeout(minuter, 1000);
    } else {
        document.body.innerHTML = "C'est fini !";
    }
    return;
};

var date_départ = new Date();
var temps_imparti = 300000;
minuter();