/*
 * Extrait les valeurs des deuxièmes propriétés des objets du tableau 'pairs'
 * et les renvoie sous forme de tableau trié.
 * 
 * Illustration :
 * pairs=[...,{"country":"Pi","continent":"Ck"},...,{"country":"Pj","continent":"Cl"},...]
 * --> [...,"Ck",...,"Cl",...]
 */

//function secondProperties(pairs)

/*
 * Renvoie un tableau d'objets possédant chacun 1 seule propriété et dont :
 * - le nom correspond à un élément de 'second_properties' (par ex. un continent)
 * - la valeur est le tableau trié des pays apparaissant dans 'pairs' et qui ont 
 * cet élément en commun (par ex., les pays du-dit continent).
 * 
 * Illustration :
 * pairs=[...,{"country":"France","continent":"Europe"},...,{"country":"Spain","continent":"Europe"},...]
 * second_properties=["Africa",...,"Europe",...,"South America"]
 * --> [...,{"Europe":[...,"France",...,"Spain",...]}, ...]
 */

// function collectFirstProperties(pairs,second_properties) 

/*
 * Identifie dans le tableau d'objets 'cps' l'objet dont la propriété (unique)
 * a même nom que 'key' et renvoie la valeur de cette propriété.
 * 
 * Illustration :
 * cps=[ ...,{"Europe":["Albania",..."Yugoslavia"]}, ...]
 * key="Europe"
 * --> ["Albania",..."Yugoslavia"]
 */ 

// function selectCountries(cps,key)

/*
 * Construit pour chaque objet O de 'pairs' un objet possédant une propriété unique
 * dont le nom est la valeur de la première propriété de O et dont la valeur
 * est la valeur de la seconde propriété de O, et renvoie ce tableau d'objets.
 * 
 * Illustration :
 * pairs=[...,{"country":"France","code":"FR"},...]
 * --> [...,{"France":"FR"},...]
 */ 

// function pairsToSingletons(pairs) 

