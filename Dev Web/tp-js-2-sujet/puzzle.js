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
        // generator.insertAuthor();

        // Q1.2 Injection des images
        // generator.insertImages();

        // Q1.3 Injection de l'énoncé
        // generator.insertStatement();

        // Injection des en-têtes du tableau (ne pas faire cette question)
        generator.insertTableHeaders();

        // Q1.4 Injection des indices
        // generator.insertHints();

        // Q1.5 Injection des menus déroulants
        // generator.insertDropDowns();

        // Q2.1 Clic sur cellules
        // generator.handleClicks();

        // Q2.2 Cochage des indices
        // generator.handleHints();

        // Q3 Gestion du formulaire
        // generator.handleDropDowns();

        return puzzle;
    })
    .catch((error) => {
        console.error('Error:', error);
    });



// Q4 Minuteur