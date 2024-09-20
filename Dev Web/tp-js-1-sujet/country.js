// 2 variables globales à modifier dans l'écouteur window.onload
var countries = {
    "names": [], // ["Afghanistan", ...]
    "codes": {}, // {"Afghanistan":"AF", ...}
    "flags": {} // {"Afghanistan":"data:image...", ...}
};

var continents = []; // [{"Asia":["Afghanistan","Armenia",...]}, ...]


window.addEventListener("load", (event) => {
    // Q1 Extraction des noms de pays à partir du tableau HTML
    // A COMPLETER <---
    // --> A COMPLETER
    let table = document.querySelector("table");
    let trs = table.querySelectorAll("tr");
    trs.forEach(function(tr) {
        let tds = tr.querySelectorAll("td");
        tds.forEach(function(td) {
                countries.names.push(td.textContent);
        });
    });
    console.log(countries.names);

    // Q2 Extraction des codes de pays du fichier country_codes.json
    fetch('country_codes.json', {
            method: 'GET'
        })
        .then((response) => response.json())
        .then((country_codes) => {
            console.log('Success:', country_codes);
            // A COMPLETER <---
            // [ ..., {"France":"FR"}, ...] --> {..., "France":"FR", ...}
            // --> A COMPLETER
            country_codes.forEach((item) => {
                let country = Object.keys(item)[0];
                let code = item[country];
                countries.codes[country] = code;
            });
                
            console.log(countries.codes);
            return countries.codes;
        })
        .catch((error) => {
            console.error('Error:', error);
        });

    // Q3 Extraction des continents de pays à partir du tableau country_continents (importé de country_continents.js) 
    // A COMPLETER <---
    // [{"country":"Afghanistan","continent":"Asia"}, ...] --> [{"Asia":["Afghanistan","Armenia",...]}, ...]
    // --> A COMPLETER
    console.log(continents);

    // Q4 Extraction des drapeaux de pays à partir de la constante country_flags (importée de country_flags.js) 
    // A COMPLETER <---
    // [ ..., {"country":"France","flag_base64":"data:..."}, ...] --> {..., "France":"data:...", ...}
    // --> A COMPLETER
    console.log(countries.flags);

    // Q5 Mise en forme CSS
    // A COMPLETER <---
    // --> A COMPLETER

});



let handleSelectors = function() {
    // Q6 Gestion du menu
    // A COMPLETER <---
    // --> A COMPLETER
}();


let handleRadios = function() {
    // Q7 gestion des boutons radio
    // A COMPLETER <---
    // --> A COMPLETER
}();

let handleHeader = function f() {
    let tds = document.querySelectorAll("td");
    tds.forEach(function(td) {
        td.addEventListener("click", function(e) {
            let country_name = e.target.id;
            if (country_name) {
                fetch('get_country_features.php', {
                        method: 'POST',
                        body: new URLSearchParams("country_name=" + country_name),
                    })
                    .then((response) => response.json())
                    .then((country) => {
                        console.log('Success:', country);
                        // Q8 clic sur cellule
                        // A COMPLETER <---
                        // --> A COMPLETER
                        return country;
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
            }
        });
    });
}();