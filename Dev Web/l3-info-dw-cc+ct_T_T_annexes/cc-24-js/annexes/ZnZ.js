/*
Chaque instance de ZnZ représente l'anneau quotient des entiers relatifs pour un `n` donné.

Les opérations arithmétiques sur les classes de l'anneau sont implémentées par des méthodes 
acceptant des entiers relatifs quelconques mais qui renvoient toutes l'unique entier 
dans `0..n-1` équivalent au résultat modulo `n`. P. ex. l'opposé de la classe de 2 dans Z/5Z
est la classe de 3 (qui est aussi celle de -2).

Les opérations implémentées sont :
- la négation `-x` : méthode `opposite`
- l'addition `x + y` : méthode `plus`
- la multiplication `x * y` : méthode `times`
- l'exponentiation `x * k` : méthode `exponentiation` où `k>=0`
- l'inversion `x**-1` qui n'est définie que pour les `x` premiers avec `n` : méthode `inverse`

La classe fournit aussi 2 méthode statiques de calcul de PGCD et de résolution du théorème des
restes chinois.
*/
class ZnZ {
    // ordre de l'anneau
    #n;

    // constructeur
    constructor(n) {
        if (!Number.isInteger(n) || n < 1) {
            throw Error(`n (${n}) doit être un entier > 0 pour construire l'anneau quotient &Zopf;/n&Zopf;`);
        }
        this.#n = n;
    }

    // accesseur à l'ordre `n` de l'anneau
    get n() {
        return this.#n;
    }

    // vérifie que x et y sont des entiers
    checkIntegrality(x, y) {
        if (!Number.isInteger(x)) {
            throw Error(`${x} n'est pas entier`);
        }
        if (!Number.isInteger(y)) {
            throw Error(`${y} n'est pas entier`);
        }
    }

    // calcule `x mod n` ou lève une exception si `x` n'est pas entier.
    // `x mod n` est le reste dans `0..n-1` de la division euclidienne de `x`par `n`.
    // Attention : l'opérateur JS du reste `x % y` n'équivaut pas à l'opération modulo `x mod y` 
    // quand `x`et `y` n'ont pas le même signe, p. ex. 
    // (-7) % 5 = -2 != 3 = (-7 mod 5) = ((-7 % 5) + 5) % 5
    // La formulation de l'opération `x mod n` par `((x % n) + n) % n` est correcte dans tous les
    // cas de figure.
    modulo(x) {
        this.checkIntegrality(x, x);
        return ((x % this.n) + this.n) % this.n;
    }

    // calcule l'opposé de `x` dans Z/nZ ou lève une exception si `x` n'est pas entier
    opposite(x) {
        this.checkIntegrality(x, x);
        return this.modulo(this.n - this.modulo(x));
    }

    // calcule la somme de `x` et `y` dans ZnZ ou lève une exception si `x` et `y` ne sont pas entiers
    plus(x, y) {
        this.checkIntegrality(x, y);
        return this.modulo(x + y);
    }

    // calcule le produit de `x`et `y` dans ZnZ ou lève une exception si `x` et `y` ne sont pas entiers
    times(x, y) {
        this.checkIntegrality(x, y);
        return this.modulo(x * y);
    }

    // calcule la puissance `k` de `x` dans Z/nZ ou lève une exception si `x` et `k` ne sont pas entiers ou si `k` est négatif
    exponentiation(x, k) {
        this.checkIntegrality(x, k);
        if (k < 0) {
            throw Error(`Exposant négatif pour exponentiation sur l'anneau quotient &Zopf;/${this.n}&Zopf;`);
        }
        if (k == 0) {
            return 1;
        }
        let r = this.modulo(x);
        while (k !== 1) {
            r = this.modulo(x * r);
            k -= 1;
        }
        return r;
    }

    // calcule la puissance `k<0` de `x` dans Z/nZ ou lève une exception si `x` n'est pas entier ou n'est pas inversible
    inverse(x, k = -1) {
        this.checkIntegrality(x, k);
        if (k > 0) {
            throw Error(`Exposant positif pour l'inversion sur l'anneau quotient &Zopf;/${this.n}&Zopf;`);
        }
        let pgcd = ZnZ.pgcd(x, this.n);
        if (pgcd["gcd"] != 1) {
            throw Error(`${x} n'est pas inversible dans &Zopf;/${this.n}&Zopf`);
        }
        return this.exponentiation(this.modulo(pgcd["cx"]), -k);
    }

    /*
    Calcule le PGCD de `x` et `y` et une paire de coefficients de Bézout `cx` et `cy`.

    Renvoie l'objet {"x": `x`, "y": `y`, "gcd": `gcd`, "cx": `cx`, "cy": `cy`} où :
    - `gcd` est le PGCD de `x` et `y`
    - l'équation `gcd = cx*x + cy*y` est satisfaite (Théorème de Bachet-Bézout).
    
    Si `x` et `y` ne sont pas des entiers, renvoie l'objet  {"x": `x`, "y": `y`, "gcd": "", "cx": "", "cy": ""}.
    
    Rappels. `x` et `y` sont co-premiers ssi `gcd=1`. Le cas échéant, `cx` est l'inverse 
    modulo `y` de `x` (identité de Bézout).
    */
    static pgcd(x, y) {
        if (!Number.isInteger(x) || !Number.isInteger(y)) {
            throw Error(`Le PGCD se calcule sur des entiers`);
        }
        // initialisation
        let r, u, v, r1, u1, v1, q;
        [r, u, v, r1, u1, v1] = [Math.abs(x), 1, 0, Math.abs(y), 0, 1];
        // les égalités r = x*u+y*v et r1 = x*u1+y*v1 sont des invariants de boucle
        while (r1 != 0) {
            q = Math.trunc(r / r1);
            [r, u, v, r1, u1, v1] = [r1, u1, v1, r - q * r1, u - q * u1, v - q * v1];
        }
        return {
            "x": x,
            "y": y,
            "gcd": r,
            "cx": u,
            "cy": v,
        }
    }

    // calcule l'unique entier modulo `n1*n2` équivalent à `a1` modulo `n1` et à `a2` modulo `n2` si
    // `n1` et `n2` sont copremiers selon le théorème des restes chinois appliqué au cas de 2 anneaux.
    //
    // Lève une exception si :
    // - l'un des arguments obligatoires n'est pas un entier positif
    // - `a1` est en dehors de la plage `0..n1`
    // - `a2` est en dehors de la plage `0..n2`
    // - `n1` est égal à `n2`
    // - `n1` et `n2` ne figurent pas dans le tableau de nombres premiers `primes` s'il est non-vide
    // NB. La méthode ne vérifie pas que `primes` contient exclusivement des nombres premiers ni, 
    // s'il est vide, que `n1` est copremier avec `n2` comme requis par le théorème.
    //
    // Renvoie un objet {"n1": `n1`, "n2": `n2`, "a1": `a1`, "a2": `a2`, "m1": `m1`, "m2": `m2`, "x": `x`} 
    // vérifiant 
    // - `x = (a1 * m2 * n2 + a2 * m1 * n1) mod n1*n2` 
    // - `pgcd(n1,n2) = m1 * n1 + m2 * n2` 
    static chineseTheorem(n1, n2, a1, a2, primes = []) {
        if (!Number.isInteger(n1) || n1 < 0) {
            throw Error(`Les restes chinois se calculent avec n1=${n1} entier positif`);
        }
        if (!Number.isInteger(n2) || n2 < 0) {
            throw Error(`Les restes chinois se calculent avec n2=${n2} entier positif`);
        }
        if (!Number.isInteger(a1) || a1 < 0 || a1 >= n1) {
            throw Error(`Les restes chinois se calculent avec a1=${a1} dans 0..n1=${n1}`);
        }
        if (!Number.isInteger(a2) || a2 < 0 || a2 >= n2) {
            throw Error(`Les restes chinois se calculent avec a2=${a2} dans 0..n2=${n2}`);
        }
        if (n1 === n2) {
            throw Error(`Les restes chinois se calculent avec n1=${n1} et n2=${n2} différents`);
        }
        if (primes.length && !(primes.includes(n1) && primes.includes(n2))) {
            throw Error(`Les restes chinois se calculent avec n1=${n1} et n2=${n2} copremiers`);
        }
        let ee = ZnZ.pgcd(n1, n2);
        return {
            "n1": n1,
            "n2": n2,
            "a1": a1,
            "a2": a2,
            "m1": ee["cx"],
            "m2": ee["cy"],
            "x": a1 * ee["cy"] * n2 + a2 * ee["cx"] * n1
        };
    }
}

export { ZnZ };