// Constructeur pour générateur HTML de puzzle
function PuzzleGenerator(puzzle) {
    this.puzzle = puzzle;
    // Camélisation du premier mot d'une chaîne : "il pleut" => "Il pleut"
    Object.getPrototypeOf(this).camelize = function(chaine) {
        return chaine[0].toUpperCase() + chaine.slice(1);
    }
}

// Injection de l'auteur
PuzzleGenerator.prototype.insertAuthor = function() {
    let p = document.querySelector("p");
    let a = document.createElement("A");
    a.href = this.puzzle.url;
    a.textContent = this.puzzle.auteur;
    p.appendChild(a);
}

// Injection des images
PuzzleGenerator.prototype.insertImages = function() {
    let images = this.puzzle.images;
    let div = document.querySelector("#images");
    images.forEach(image => {
        let img = document.createElement("IMG");
        img.alt = image.alt;
        img.height = "50";
        img.width = "30";
        div.appendChild(img);
        let request = new Request("img/" + image.src);
        fetch(request)
            .then(function(response) {
                if (!response.ok) {
                    throw new Error('HTTP error - status = ' + response.status);
                }
                return response.blob();
            })
            .then(function(blob) {
                let objectURL = URL.createObjectURL(blob);
                img.src = objectURL;
            })
            .catch(function(error) {
                console.error(error.message);
            });
    });
}

// Injection de l'énoncé
PuzzleGenerator.prototype.insertStatement = function() {
    let h3 = document.querySelector("h3");
    let énoncé = this.puzzle.énoncé;
    h3.textContent = énoncé;
}

// Injection des indices
PuzzleGenerator.prototype.insertHints = function() {
    let ol = document.querySelector("ol");
    let indices = this.puzzle.indices;
    indices.forEach((indice, k) => {
        let li = document.createElement("LI");
        let input = document.createElement("INPUT");
        input.type = "checkbox";
        input.name = "indices[]";
        input.id = "indice" + (k + 1);
        input.value = input.id;
        let label = document.createElement("LABEL");
        label.for = input.id;
        label.textContent = indice;
        li.appendChild(input);
        li.appendChild(label);
        ol.appendChild(li)
    });
}

// Injection des en-têtes du tableau
PuzzleGenerator.prototype.insertTableHeaders = function() {
    let aide = document.querySelector("TABLE");
    let facettes = this.puzzle.facettes;
    Array.from(aide.rows).forEach((row, i) => {
        if (i == 0) {
            Array.from(row.cells).forEach((cell, j) => {
                if (j > 0) {
                    cell.textContent = this.camelize(facettes[j].nom);
                }
            });
        } else {
            if (i == 1) {
                Array.from(row.cells).forEach((cell, j) => {
                    cell.textContent = this.camelize(facettes[1 + Math.floor(j / 3)].valeurs[j % 3]);
                });
            } else {
                let facette_id = [0, 3, 2, 1][Math.floor((i - 2) / 3)];
                Array.from(row.cells).forEach((cell, j) => {
                    if (i % 3 == 2) {
                        if (j == 0) {
                            cell.textContent = this.camelize(facettes[facette_id].nom);
                        } else
                        if (j == 1) {
                            cell.textContent = this.camelize(facettes[facette_id].valeurs[(i - 2) % 3]);
                        }
                    } else
                    if (j == 0) {
                        cell.textContent = this.camelize(facettes[facette_id].valeurs[(i - 2) % 3]);
                    }
                });
            }
        }
    });
}

// Injection des menus déroulants
PuzzleGenerator.prototype.insertDropDowns = function() {
    let facettes = this.puzzle.facettes;
    let table = document.querySelectorAll("TABLE")[1];
    Array.from(table.rows).forEach((row, i) => {
        if (i == 0) {
            Array.from(row.cells).forEach((cell, j) => {
                let facette = facettes[j].nom;
                cell.textContent = facette[0].toUpperCase() + facette.slice(1);
            });
        } else {
            Array.from(row.cells).forEach((cell, j) => {
                if (j == 0) {
                    cell.textContent = facettes[0].valeurs[i - 1];
                } else {
                    let select = document.createElement("SELECT");
                    select.name = facettes[0]["valeurs"][i - 1] + "_" + facettes[j].nom;
                    // option par défaut
                    let option = document.createElement("OPTION");
                    option.value = "vide";
                    option.selected = true;
                    option.textContent = "";
                    select.appendChild(option);
                    // options par facette j
                    valeurs = facettes[j].valeurs;
                    valeurs.forEach((valeur, k) => {
                        let option = document.createElement("OPTION");
                        option.value = valeur.replace(/-.*$/, "");
                        option.textContent = valeur;
                        select.appendChild(option);
                    });
                    cell.appendChild(select);
                }
            });
        }
    });
}

// Clic sur cellules
PuzzleGenerator.prototype.handleClicks = function() {
    document.querySelector("#aide table").addEventListener("click", function(e) {
        let td = e.target;
        if (td.nodeName == "TD") {
            if (td.textContent === "") {
                td.textContent = "X";
                td.style.backgroundColor = "Red";
            } else {
                if (td.textContent === "X") {
                    td.textContent = "O";
                    td.style.backgroundColor = "Green";
                } else {
                    if (td.textContent === "O") {
                        td.textContent = "";
                        td.style.backgroundColor = "transparent";
                    }
                }
            }
        }
    });
}

// Cochage des indices
PuzzleGenerator.prototype.handleHints = function() {
    document.querySelector("ol").addEventListener("click", function(e) {
        let input = e.target;
        if (input.nodeName == "INPUT") {
            if (input.checked) {
                input.nextElementSibling.style.textDecoration = "line-through";
            } else {
                input.nextElementSibling.style.textDecoration = "none";
            }
        }
    });
}

// Gestion du formulaire
PuzzleGenerator.prototype.handleDropDowns = function() {
    document.querySelector("form").addEventListener("submit", function(e) {
        e.preventDefault();
        // préparation des données à soumettre
        let query = new URLSearchParams();
        let proposal = {};
        puzzle.facettes[0].valeurs.forEach(function(v) {
            proposal[v] = {};
            proposal[v][puzzle.facettes[1].nom] = "";
            proposal[v][puzzle.facettes[2].nom] = "";
            proposal[v][puzzle.facettes[3].nom] = "";
        });
        document.querySelectorAll("select").forEach((select, k) => {
            let option = select.options[select.selectedIndex];
            let i = Math.floor(k / 3);
            let v = puzzle.facettes[0].valeurs[i];
            let j = k % 3;
            let w = puzzle.facettes[j + 1].nom;
            proposal[v][w] = option.value;
            query.append(select.name, option.value);
        });
        console.log(proposal);
        // requête
        fetch('puzzle.php', {
                method: 'POST', // or 'PUT'
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: query,
            })
            .then((response) => response.json())
            .then((feedback) => {
                for (const [selectName, truth] of Object.entries(feedback["réponse"])) {
                    let select = document.querySelector(`select[name=${selectName}]`);
                    if (truth) {
                        select.style.backgroundColor = "green";
                    } else {
                        select.style.backgroundColor = "red";
                    }
                }
                if (feedback["résolution"]) {
                    alert("BRAVO : énigme résolue !");
                }
            })
            .catch((error) => {
                alert(error);
            });
        /** */
    });
}