import {
    _points
} from "./échantillonneur_proto.js";

export class Echantillon {
    #f; // La fonction (callback) à échantillonner
    #n; // Le nombre de points de l'échantillon
    #minMaxX; // Abscisses min (u) et max (v) des points de l'échantillon : minMaxX = {"min":u, "max": v}

    constructor(f, n, minMaxX) {
        this.#f = f;
        this.#n = n;
        this.#minMaxX = minMaxX;
    }

    get f() { return this.#f; }
    get n() { return this.#n; }
    get minMaxX() { return this.#minMaxX; }

    /*
    Génère et renvoie le tableau de points [[x,f(x)] | i=1..n, x=Echantillon.abscisse(i, n, minMaxX)].
    Selon la fonction f et son domaine de définition (p. ex. sqrt), l'appel de f
    sur des valeurs hors du domaine pourra renvoyer NaN ou +/-Infinity.
    Le code appelant devra gérer ces points.
    */
    // REMPLACER LA FONCTION `_points` PAR UNE FONCTION ANONYME QUE VOUS IMPLEMENTEREZ :  `points = function() {...};`
    points = function() {
        let points = [];
        for (let i = 1; i <= this.#n; i++) {
            let x = Echantillon.abscisse(i, this.#n-1, this.#minMaxX);
            let fx = this.#f(x);
            if (!isNaN(fx) && isFinite(fx)) {
                points.push([x, fx]);
            }
            else {
                break;
            }
        }
        points.forEach(p => console.log(p));
        return points;
    };
    
    
    /*
    Renvoie l'abscisse du k-ième point de sorte que :
    - abscisse(1,n,Mx) == minMaxX.min et 
    - abscisse(n,n,Mx) == minMaxX.max
    */
    static abscisse(k, n, minMaxX) {
        if (minMaxX.max * minMaxX.min < 0) {
            return minMaxX.min + (((k - 1) * (Math.abs(minMaxX.max - minMaxX.min))) / (n - 1));
        } else {
            return minMaxX.min + (((k - 1) * (Math.abs(minMaxX.max + minMaxX.min))) / (n - 1));
        }
    }

}