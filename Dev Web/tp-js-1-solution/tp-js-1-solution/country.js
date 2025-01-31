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
    let tds = document.querySelectorAll("td");
    countries.names = Array.from(tds).map(td => td.innerHTML);
    // --> A COMPLETER
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
            country_codes.forEach(function(countcode) {
                countries.codes[Object.entries(countcode)[0][0]] = Object.entries(countcode)[0][1];
            });
            // --> A COMPLETER
            console.log(countries.codes);
            return countries.codes;
        })
        .catch((error) => {
            console.error('Error:', error);
        });

    // Q3 Extraction des continents de pays à partir du tableau country_continents (importé de country_continents.js) 
    // A COMPLETER <---
    // [{"country":"Afghanistan","continent":"Asia"}, ...] --> [{"Asia":["Afghanistan","Armenia",...]}, ...]
    let continent_set = new Set(country_continents.map(councont => councont["continent"]));
    let continent_tab = Array.from(continent_set).sort();
    // -> [...,{"Europe":[...,"France",...,"Spain",...]}, ...]
    continents = continent_tab.map(function(cont) {
        let o = {};
        // {"Europe":[...,"France",...,"Spain",...]}
        o[cont] = country_continents.filter(councont => councont["continent"] == cont).map(councont => councont["country"]);
        o[cont].sort((a, b) => a[0] > b[0]);
        return o;
    });
    // --> A COMPLETER
    console.log(continents);

    // Q4 Extraction des drapeaux de pays à partir de la constante country_flags (importée de country_flags.js) 
    // A COMPLETER <---
    // [ ..., {"country":"France","flag_base64":"data:..."}, ...] --> {..., "France":"data:...", ...}
    country_flags.forEach(function(countflag) {
        countries.flags[Object.entries(countflag)[0][1]] = Object.entries(countflag)[1][1];
    });
    // --> A COMPLETER
    console.log(countries.flags);

    // Q5 Mise en forme CSS
    // A COMPLETER <---
    let elements = document.querySelectorAll("table,tr,td");
    elements.forEach((elt) => {
        elt.style.textAlign = "center"
    });
    elements.forEach((elt) => {
        elt.style.fontSize = "75%"
    });
    let div2 = document.querySelector("body>div:nth-child(2)");
    div2.classList.add("row");
    div2.querySelector("div:first-child").classList.add("side");
    // --> A COMPLETER

});



let handleSelectors = function() {
    // Q6 Gestion du menu
    // A COMPLETER <---
    function display(names) {
        let tds = document.querySelectorAll("td");
        tds.forEach(function(td) {
            if (names.includes(td.id)) {
                td.style.visibility = "visible";
            } else {
                td.style.visibility = "hidden";
            }
        });
    }

    let selector = document.querySelector("#continents");
    selector.addEventListener("change", function(e) {
        let selector = e.target;
        let option = selector.options[selector.selectedIndex].value;
        if (option !== "all") {
            let continent_countries = continents.filter(cont => Object.keys(cont)[0] === option);
            display(continent_countries[0][option]);
        } else {
            display(countries.names);
        }
    });
    // --> A COMPLETER
}();


let handleRadios = function() {
    // Q7 gestion des boutons radio
    // A COMPLETER <---
    function display(data, coche) {
        let tds = document.querySelectorAll("td");
        if (coche === "codes") {
            tds.forEach(function(td) {
                td.innerHTML = data.codes[td.id];
                console.log(td.id);
            });
        } else
        if (coche === "noms") {
            tds.forEach(function(td) {
                td.innerHTML = td.id;
            });
        } else {
            tds.forEach(function(td) {
                td.innerHTML = "";
                let img = document.createElement("img");
                img.alt = td.id;
                img.src = (data.flags[td.id] || " ");
                img.className = "flag";
                td.appendChild(img);
            });
        }
    }

    let radios = document.querySelectorAll("input[type='radio']");
    radios.forEach(function(radio) {
        radio.addEventListener("click", function(e) {
            var coche = Array.from(document.querySelectorAll("input[name=pays]")).filter((e) => e.checked)[0].value;
            display(countries, coche);
        });
    });
    // --> A COMPLETER
}();

let handleHeader = function f() {
    let tds = document.querySelectorAll("td");
    tds.forEach(function(td) {
        td.addEventListener("click", function(e) {
            let country_name = e.target.id;
            if (country_name) {
                fetch('country_features.php', {
                        method: 'POST',
                        body: new URLSearchParams("country_name=" + country_name),
                    })
                    .then((response) => response.json())
                    .then((country) => {
                        console.log('Success:', country);
                        // Q8 clic sur cellule
                        // A COMPLETER <---
                        let h = document.querySelector(".header");
                        let gov = country.government;
                        let lif = country.expectancy;
                        h.innerHTML = country_name + "<br/>" +
                            "life expectancy : " + (lif !== null ? lif : "?") + " - " +
                            "government type : " + (gov !== null ? gov : "?");
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