let data = function() {
    let countries = country_continents.map(pc => pc["country"]);
    let continents = secondProperties(country_continents);
    let country_by_continent = collectFirstProperties(country_continents,continents);
    let codes = pairsToSingletons(country_codes);
    let flags = pairsToSingletons(country_flags);
    return {
        "countries":countries,
        "country_by_continent":country_by_continent,
        "codes":codes,
        "flags":flags,
    };
}();

//Q5 gestion du menu
let handleSelectors = function() {
    // A COMPLETER
}();

//Q6 gestion de l'entête par requête Ajax
let handleHeader = function f() {
    // A COMPLETER
}();