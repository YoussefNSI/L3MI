import {
    Echantillon
} from "./échantillonneur.js";

// Trace la grille
export function tracerGrille() {
    // paramétrage du contexte
    this.contexte.strokeStyle = 'black';
    this.contexte.lineWidth = 0.05;
    this.contexte.setLineDash([3, 3]);
    this.contexte.beginPath();

    // grille verticale
    let uv = this.transformer(this.repère.X.gauche[0], this.repère.X.gauche[1]);
    this.contexte.moveTo(uv.X, uv.Y);
    for (let x = -this.maxXY.X; x <= this.maxXY.X; x += 2 * this.maxXY.X / 10) {
        let uv1 = this.transformer(x, -this.maxXY.Y);
        let uv2 = this.transformer(x, this.maxXY.Y);
        this.contexte.moveTo(uv1.X, uv1.Y);
        this.contexte.lineTo(uv2.X, uv2.Y);
        this.contexte.stroke();
    }

    // grille horizontale
    uv = this.transformer(this.repère.Y.bas[0], this.repère.Y.bas[1]);
    this.contexte.moveTo(uv.X, uv.Y);
    for (let y = -this.maxXY.Y; y <= this.maxXY.Y; y += 2 * this.maxXY.Y / 10) {
        let uv1 = this.transformer(-this.maxXY.X, y);
        let uv2 = this.transformer(this.maxXY.X, y);
        this.contexte.moveTo(uv1.X, uv1.Y);
        this.contexte.lineTo(uv2.X, uv2.Y);
        this.contexte.stroke();
    }
};


/*
    Calcule le transformé (u,v) sur le canevas d'un point (x,y) et le renvoie au format {"X": u, "Y": v} par 
    - dimensionnement en appliquant les coefficients de this.rapportXY
    - et décalage par rapport au centre du canevas.
    Emet un message d'avertissement en console et renvoie {"X": false, "Y": false} pour tout point (x,y) hors-limite,
    cad. si l'une au moins de ses coordonnées vaut NaN ou +/-Infinity ou si |x| > this.maxXY.X ou |y| > this.maxXY.Y.
    */
export function transformer(x, y) {
    if (!isFinite(x) || !isFinite(y)) {
        return {
            "X": false,
            "Y": false
        };
    }

    if (Math.abs(x) > this.maxXY.X) {
        console.warn(Math.abs(x) + " > " + this.maxXY.X);
        return {
            "X": false,
            "Y": false
        };
    }

    if (Math.abs(y) > this.maxXY.Y) {
        console.warn(Math.abs(y) + " > " + this.maxXY.Y);
        return {
            "X": false,
            "Y": false
        };
    }

    let u = x * this.rapportXY.X;
    let v = y * this.rapportXY.Y;
    console.debug(`(${u},${v})`);

    return {
        "X": this.repère.X.centre[0] + u,
        "Y": this.repère.X.centre[1] - v
    };
};


/*
    - Transforme le tableau P=[...,[u,v],...] de n points passé en paramètre en un tableau Q de n transformés 
    (positions sur le canevas) par appel à la méthode `this.transformer`.
    - Trace la courbe correspondante à Q en reliant les ordonnées des transformés successifs Q[i].Y et Q[i+1].Y) 
    par des segments (1<=i<n).    
    Remarques :
    - Le tracé utilise la couleur CSS `strokeStyle`.
    - Une exception est levée si P ne vérifie pas P[i].X < P[i+1].X (1<=i<n).
    - Les points hors limite sont omis du tracé et donne lieu à un avertissement en console.
*/
export function tracer(P, strokeStyle) {
    this.contexte.beginPath();
    this.contexte.strokeStyle = strokeStyle;
    this.contexte.setLineDash([]);
    this.contexte.lineWidth = 1;
    P.forEach((xy, k) => {
        if (k > 1 && P[k - 1].X >= P[k].X) {
            throw new Error(P, "série de points mal formée");
        }

        let uv = this.transformer(xy[0], xy[1]);
        if (uv.X === false || uv.Y === false) {
            console.warn("point en dehors des bornes fixées");
        } else if (k == 0) {
            this.contexte.moveTo(uv.X, uv.Y);
        } else {
            this.contexte.lineTo(uv.X, uv.Y);
            this.contexte.stroke();
        }
    });
};


 /* 
 Procède en 3 étapes :
 (1) Génère pour la fonction réelle décrite par `meta_f` un échantillon de `n` points 
         dont les abscisses sont répartis à intervalles réguliers dans l'intervalle [-this.maxXY.X,this.maxXY.X]
 (2) Trace la courbe échantillonnée sur le canevas selon la fréquence renseignée par le visiteur
 (3) Ajoute le descripteur `meta_f` au tableau `log` et ajoute une ligne au tableau HTML 
 contenant la formulation algébrique de la fonction.

 Le descripteur `meta_f` est au format
 {
     "type": chaîne dans {"linéaire", "exponentiation", "racine", "e", "logarithme", "sinus"}
     "f": fonction de rappel JS implémentant la fonction décrite
     "paramètres": tableau (potentiellement vide) des paramètres de la fonction dans l'ordre des champs HTML correspondants
     "strokeStyle": code couleur CSS pour tracer et tabuler la fonction (p. ex. "orange" ou un code RGB "rgb(5,200,89)")
 }

 Remarques :
 - L'échantillonnage s'effectue par invocation de la méthode `points` sur une instance d'`Echantillon`.
 - Si la fréquence de tracé F (renseignée dans le champ du formulaire) est > 0, 
     le tracé est non-bloquant (cad. asynchrone en utilisant `setInterval`) et chaque segment de la courbe 
     est tracé toutes les F/10 secondes en commençant par le point le plus à gauche.
 - La ligne (mono-cellule) à ajouter au tableau HTML contiendra une chaîne de caractères construite à partir
     du type de la fonction et de la valeur de ses arguments (p. ex. "2x + 1", "3 sin(2Pix/4 + 1.07").
 - La couleur `meta_f.strokeStyle` est utlisée pour le tracé et comme couleur de fond de la ligne HTML.
 */

export function dessiner(n, meta_f, log) {
    // Absisses min et max
    let minMaxX = {
        "min": -this.maxXY.X,
        "max": this.maxXY.X
    };
    // Génération d'un échantillon de n points de la fonction f
    let échantillon = new Echantillon(meta_f.f, n, minMaxX);
    let P = échantillon.points();
    console.debug("P", P);

    // Tracé de l'échantillon
    let fréquence = parseInt(document.querySelector('input[name="échantillon_f"]').value) * 100;
    if (fréquence) {
        let k = 0;
        const relier2Points = function() {
            // console.debug("this", this);
            // console.debug("k", k);
            if (k + 1 < P.length) {
                this.tracer(P.slice(k, k + 2), meta_f.strokeStyle);
                k += 1;
            }
        }.bind(this);
        setInterval(relier2Points, fréquence);
    } else {
        this.tracer(P, meta_f.strokeStyle);
    }

    // Log de la fonction et affichage dans le tableau
    log.push(meta_f);
    let tableau = document.querySelector('table');
    let ligne = document.createElement('TR');
    let cellule = document.createElement('TD');

    switch (meta_f.type) {
        case "linéaire":
            let al = meta_f.paramètres[0];
            let bl = meta_f.paramètres[1];
            cellule.innerHTML = `${al}x + ${bl}`;
            break;
        case "exponentiation":
            let ne = meta_f.paramètres[0];
            cellule.innerHTML = `x<sup>${ne}</sup>`;
            break;
        case "racine":
            let nr = meta_f.paramètres[0];
            cellule.innerHTML = `x<sup>1/${nr}</sup>`;
            break;
        case "e":
            cellule.innerHTML = "e<sup>x</sup>";
            break;
        case "logarithme":
            let blog = meta_f.paramètres[0];
            cellule.innerHTML = `log<sub>${blog}</sub>(x)`;
            break;
        case "sinus":
            let As = meta_f.paramètres[0];
            let ns = meta_f.paramètres[1];
            let phis = meta_f.paramètres[2];
            cellule.innerHTML = `${As} sin(2&Pi;x/${ns} + ${phis})`;
            break;
        default:
            break;
    }
    console.table(log);
    cellule.style.backgroundColor = meta_f.strokeStyle;
    ligne.appendChild(cellule);
    tableau.appendChild(ligne);
};