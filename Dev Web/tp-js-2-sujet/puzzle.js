let puzzle = null;

fetch("puzzle.json", {
  method: "GET",
})
  .then((response) => response.json())
  .then((puzzles) => {
    // Extraction du premier puzzle et construction du générateur
    console.log("Success:", puzzles);
    puzzle = puzzles[0];
    let generator = new PuzzleGenerator(puzzle);

    // Q1.1 Injection de l'auteur
    // generator.insertAuthor();
    entete = document.querySelector("body");
    entete.querySelector("p").innerHTML +=
      "<a href=" + puzzle["url"] + ">" + puzzle["auteur"] + " </a>";

    // Q1.2 Injection des images
    // generator.insertImages();
    puzzle["images"].forEach((image) => {
      console.log(image);
      var requete = new Request("img/" + image["src"], { method: "GET" });
      fetch(requete).then((response) => {
        if (response.ok) {
          async function handleResponse(response) {
            try {
              const blob = await response.blob();
              const url = URL.createObjectURL(blob);
              console.log(url);
              document.getElementById("images").innerHTML +=
                "<img src=" +
                url +
                " alt=" +
                image[1] +
                " width='30' height='50' " +
                "></img>";
            } catch (erreur) {
              console.error("Erreur lors de la création de l'objet:", erreur);
            }
          }

          handleResponse(response);
        } else {
          console.log(
            "Erreur de chargement de l'image. Code d'erreur : " +
              response.status
          );
        }
      });
    });

    // Q1.3 Injection de l'énoncé
    // generator.insertStatement();
    enonce = puzzle["énoncé"];
    entete.querySelectorAll("div")[1].innerHTML += "<p>" + enonce + "</p>";

    // Injection des en-têtes du tableau (ne pas faire cette question)
    generator.insertTableHeaders();

    // Q1.4 Injection des indices
    // generator.insertHints();
    indices = puzzle["indices"];
    i = 1;
    indices.forEach((indice) => {
      console.log(indice);
      entete.querySelectorAll("div")[3].innerHTML +=
        "<li><input type='checkbox' name='indices[]' id='indice" +
        i +
        "'><label> " +
        indice +
        "</label></li>";
      i++;
    });

    // Q1.5 Injection des menus déroulants
    // generator.insertDropDowns();
    fieldset = entete.querySelectorAll("div")[4].querySelector("fieldset");
    table = fieldset.querySelector("table");

    facette = puzzle["facettes"];
    j = 0;
    i = 1;
    noms = false;
    facette.forEach((face) => {
        table.querySelectorAll("tr")[0].querySelectorAll("th")[j].innerHTML += face["nom"];
        if(!noms){
            let i=1;
            face['valeurs'].forEach((valeur) => {
                table.querySelectorAll("tr")[i].querySelectorAll("td")[j].innerHTML = valeur;
                i++;
            });
            noms = true;
        }
        else{
            let i=1;
            table.querySelectorAll("tr")[i].querySelectorAll("td")[j].innerHTML = "<select>";
            face['valeurs'].forEach((valeur) => {
                table.querySelectorAll("tr")[i].querySelectorAll("td")[j].innerHTML = "<option value=" +valeur+"</option>";
                i++;
            });
            
        }
      j++;
    });

    // Q2.1 Clic sur cellules
    // generator.handleClicks();

    // Q2.2 Cochage des indices
    // generator.handleHints();

    // Q3 Gestion du formulaire
    // generator.handleDropDowns();

    return puzzle;
  })
  .catch((error) => {
    console.error("Error:", error);
  });

// Q4 Minuteur
