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
    points = _points;

    // points = function () {
    //     let indices = Array(this.#n).keys();
    //     let X = [...indices].map(k => Echantillon.abscisse(k + 1, this.#n, this.#minMaxX));
    //     let F = X.map(v => [v, this.#f(v)]);
    //     console.table(F);
    //     return F;
    // } // OU points.call(this); SI UTILISATION DU MODULE IMPORTE

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