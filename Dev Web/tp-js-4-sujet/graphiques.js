import {
    p_températures_départementales
}
from "./températures_départementales.js";

import {
    p_températures_anjou
}
from "./températures_anjou.js";

import {
    p_communes
}
from "./communes.js";

/*
 Promesse produisant le graphique des températures moyennes quotidiennes sur 
 2018-2023 pour les départements 49, 59, 65 et 83.
 les données proviennent de la réponse de la promesse `p_températures_départementales` (voir import).
 Le graphique est construit avec la librairie Plot en invoquant la méthode 
 Plot.plot() dont l'argument sera au format :
    {
        title: "Températures moyennes quotidiennes",
        subtitle: "Départements 49, 59, 65, 83 - Période 2018-2023",
        color: { legend: true },
        marks: [    Plot.ruleY([-10, 35]),
                    Plot.lineY(relevés["49"], {x: "date", y: "tmoy", stroke: "département"}),
                    ...
                    Plot.lineY(relevés["83"], {x: "date", y: "tmoy", stroke: "département"})
                ]
    }
où `relevés` est le résultat de la promesse `p_températures_départementales`.

Le graphique est un élément HTML contenant un titre, un sous-titre, une légende et
le graphique à proprement parler (élément SVG) qui est à insérer dans le conteneur
de classe "températures_4D".
 */
p_températures_départementales
    .then(relevés => {
        // marques du graphique
        let marques = [Plot.ruleY([-10, 35])];
        Object.entries(relevés).map((code_relevés) =>
            marques.push(
                Plot.lineY(code_relevés[1], {
                    x: "date",
                    y: "tmoy",
                    stroke: "département"
                })
            )
        );
        console.debug("marques", marques);

        // tracé
        const plot = Plot.plot({
            title: "Températures moyennes quotidiennes",
            subtitle: "Départements 49, 59, 65, 83 - Période 2018-2023",
            //caption: "Figure 1.",
            color: {
                legend: true
            },
            marks: marques
        });

        console.log(plot)
        // affichage
        const div = document.querySelector(".températures_4D");
        div.append(plot);
    })
    .catch(e => console.log(e.message));


/*
 Promesse produisant le graphique des températures minimales/moyennes/maximales 
 pour le département 49 sur 2022.
 Les données proviennent de la promesse `p_températures_anjou` (voir import).
 Le graphique est construit avec la librairie Plot en invoquant la méthode 
 Plot.plot() dont l'argument sera au format :
     {
        title: "Températures mensuelles angevines en 2022",
        color: { legend: true },
        marks: [    Plot.ruleY(Tmin - 5, Tmax + 5]),
                    Plot.dot(relevés, {x: "date", y: "température", stroke: "classe"})
                ]
    }
où `relevés` est le résultat de la promesse `p_températures_anjou`
et `Tmin` et `Tmax` sont respectivement le minimum et maximum des températures 
minimales et maximales des relevés angevins.
Le graphique est à insérer dans le conteneur de classe "températures_anjou".
 */
p_températures_anjou
    .then(relevés => {
        // A COMPLETER
    }).catch(erreur => {
        console.error(erreur.message);
    });


/*
 Ecouteur générant le graphique des communes d'une région pour chaque région
 sélectionnée.
 Les données proviennent de la promesse renvoyée par l'appel de `p_communes`
 avec le code de la région (voir import).
 Le graphique est construit avec la librairie Plot en invoquant la méthode 
 Plot.plot() dont l'argument sera au format :
     {
        axis: null,
        width: 800,
        height: 10 * communes.length,
        margin: 10,
        marginLeft: 200,
        marginRight: 200,
        marks: [Plot.tree(communes, {textStroke: "white"})]
    }
où `communes` est un tableau des chaînes, chacune correspondant à une commune
de la région et formatté comme suit : "nom_région/nom_département/nom_commune".
Le graphique est à insérer dans le conteneur de classe "communes".
 */

document.querySelector("select").addEventListener("change", function(e) {
    let région = e.target.options[e.target.selectedIndex].value;
    if (région === "0") {
        return 0;
    }

    p_communes(région)
        .then(communes_région => {
            // A COMPLETER
        }).catch(erreur => {
            console.error(erreur.message);
        });
});