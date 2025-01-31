/*
    Renvoie le tableau de points [[x,f(x)] | i=1..n, x=Echantillon.abscisse(i,n,minMaxX)].
    Selon la fonction f et son domaine de définition (p. ex. sqrt), l'appel de f
    pour une valeur hors du domaine pourra renvoyer NaN ou +/-Infinity.
*/
export function _points() {
    let indices = Array(this.n).keys();
    let X = [...indices].map(k => this.constructor.abscisse(k + 1, this.n, this.minMaxX));
    let F = X.map(v => [v, this.f(v)]);
    console.table(F);
    return F;
}