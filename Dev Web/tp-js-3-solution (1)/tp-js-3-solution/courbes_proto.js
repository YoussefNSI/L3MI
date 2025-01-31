// Imports
import {
    Traceur
} from "./traceur.js";


//////////////////////////////////////////////////////////////////////////////////////////
/*
    Met en place un écouteur sur les cases à cocher et si une case est cochée :
    - (1) construit le descripteur de fonction à partir des paramètres renseignés dans le formulaire
            (voir la fonction de test ou la méthode Traceur.dessiner pour le format de descripteur attendu)
            et en générant un code RGB aléatoire à l'aide de la fonction `rgb()`
    - (2) dessine la fonction en invoquant la méthode `dessiner` sur l'objet `traceur`en lui passant
            la taille de l'échantillon renseignée dans le formulaire, le descripteur et le log passé en argument
            (voir la méthode Traceur.dessiner)
    - (3) laisse la case cochée pendant 2 secondes avant de la décocher.
*/
export function tracer_(traceur, log) {
    let checkboxes = document.querySelectorAll("[name='fonction[]']");
    checkboxes.forEach(
        c => c.addEventListener("click",
            e => {
                let n = parseInt(document.querySelector("input[name='échantillon_n']").value);

                if (c.checked) {
                    let meta_f = {
                        "type": "",
                        "f": null,
                        "paramètres": [],
                        "strokeStyle": rgb_()
                    };

                    switch (c.value) {
                        case "linéaire":
                            let al = parseInt(document.getElementsByName("linéaire_a")[0].value);
                            let bl = parseInt(document.getElementsByName("linéaire_b")[0].value);
                            meta_f.type = "linéaire";
                            meta_f.f = function(x) {
                                return al * x + bl;
                            };
                            meta_f.paramètres = [al, bl];
                            break;
                        case "exponentiation":
                            let ne = parseInt(document.getElementsByName("exponentiation_n")[0].value);
                            meta_f.type = "exponentiation";
                            meta_f.f = function(x) {
                                return Math.pow(x, ne);
                            };
                            meta_f.paramètres = [ne];
                            break;
                        case "racine":
                            let nr = parseInt(document.getElementsByName("racine_n")[0].value);
                            meta_f.type = "racine";
                            meta_f.f = function(x) {
                                return Math.pow(x, 1 / nr);
                            };
                            meta_f.paramètres = [nr];
                            break;
                        case "e":
                            meta_f.type = "e";
                            meta_f.f = function(x) {
                                return Math.exp(x);
                            };
                            break;
                        case "logarithme":
                            let blog = document.getElementsByName("logarithme_b")[0].value;
                            blog = (blog === "e") ? Math.E : parseFloat(blog);
                            meta_f.type = "logarithme";
                            meta_f.f = function(x) {
                                return Math.log(x) / Math.log(blog);
                            };
                            meta_f.paramètres = [blog];
                            break;
                        case "sinus":
                            let As = parseFloat(document.getElementsByName("sinus_A")[0].value);
                            let ns = parseFloat(document.getElementsByName("sinus_n")[0].value);
                            let phis = parseFloat(document.getElementsByName("sinus_phi")[0].value);
                            meta_f.type = "sinus";
                            meta_f.f = function(x) {
                                return As * Math.sin(2 * Math.PI * x / ns + phis);
                            };
                            meta_f.paramètres = [As, ns, phis];
                            break;
                        default:
                            break;
                    }

                    traceur.dessiner(n, meta_f, log);

                    c.disabled = true;
                    setTimeout(() => {
                        c.checked = false;
                        c.disabled = false;
                    }, 2000);
                }
            })
    )
}

// Génère un code rgb aléatoire sous forme de chaîne de caractères, p. ex. "rgb(10,240,89)"
function rgb_() {
    return `rgb(${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)}, ${Math.floor(Math.random() * 255)})`;
}


//////////////////////////////////////////////////////////////////////////////////////////
/*
    Met en place un écouteur sur les clics du bouton de soumission qui, le cas échéant,
    (1) supprime le canevas existant et le remplace par un canevas de même dimension
    (2) construit une instance de Traceur pour ce canevas selon les coordonnées max renseignées dans le formulaire
    (3) trace le repère et éventuellement la grille si le bouton radio a été coché
    (4) redessine les courbes de toutes les fonctions figurant dans le log 
*/
export function regénérerTracés_(log) {
    let submit = document.querySelector("button[type='submit']");
    submit.addEventListener("click", e => {
        // remplace le canevas
        let canevas = document.querySelector('canvas');
        //contexte.clearRect(0, 0, contexte.canvas.width, contexte.canvas.height);
        let canevas1 = document.createElement("canvas");
        canevas1.width = canevas.width;
        canevas1.height = canevas.height;
        canevas.parentElement.appendChild(canevas1);
        canevas.parentElement.removeChild(canevas);
        canevas = canevas1;

        // construit un Traceur selon les coordonnées max renseignées dans le formulaire
        let contexte = canevas.getContext("2d");
        let marge = {
            "X": 20,
            "Y": 20
        };
        let maxXY = {
            "X": parseInt(document.querySelector("input[name='max_x']").value),
            "Y": parseInt(document.querySelector("input[name='max_y']").value)
        };
        let traceur = new Traceur(canevas, contexte, marge, maxXY);

        // trace le repère et éventuellement la grille si le bouton radio a été coché
        traceur.tracerRepère();
        let grille = document.querySelector("input[name='grille']");
        grille.checked ? traceur.tracerGrille() : null;

        // redessine les courbes de toutes les fonctions figurant dans le log 
        let n = parseInt(document.querySelector("input[name='échantillon_n']").value);
        document.querySelector('table').innerHTML = "";
        let m = log.length;
        for (let i = 0; i < m; ++i) {
            console.log("log", log);
            let meta_f = log.shift();
            console.log("meta_f", meta_f);
            traceur.dessiner(n, meta_f, log);
        }
    });
};