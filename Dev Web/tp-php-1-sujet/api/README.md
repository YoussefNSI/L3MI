# ROUTES

Deux routes sont à implémenter, chacune associées à un ou plusieurs points de terminaison (alias endpoints).
La première route permet d'obtenir, de créer, de mettre à jour ou de supprimer un continent, de lister les continents ou de rechercher un continent.
La seconde route permet d'obtenir, de créer, de mettre à jour ou de supprimer un pays d'un continent, de lister les pays d'un continent ou de rechercher un pays de continent.


## VERBES HTTP

Les endpoints utilisent les verbes HTTP suivants :
- lecture : GET
- création : POST
- mise à jour : POST (au lieu de PATCH ou PUT)
- suppression : DELETE
- listing : GET
- recherche : GET
- information : GET


## FORMAT ET REECRITURE D'URL

[Chaque endpoint est donné par son URL "propre" et l'URL obtenue après application de règles de ré-écriture](https://www.webrankinfo.com/dossiers/techniques/tutoriel-url-rewriting).

[Etant donné que tout pays appartient à un continent (relation n-1), le choix de conception de l'API est d'imbriquer les routes et endpoints associés](https://www.moesif.com/blog/technical/api-design/REST-API-Design-Best-Practices-for-Sub-and-Nested-Resources/).

Les endpoints de la route des pays sont donc préfixés par celle des continents.


# ENDPOINTS


## LISTE DES ENDPOINTS

|-----------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------------|
|   OPERATION                       |   URL PROPRE                                      |   URL REECRITE                                                                                  |
|-----------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------------|
|   Liste les continents            |   GET /{version}/continents                       |   GET /router.php?version={version}&controller=continents&method=list                           |
|   Accède à un continent           |   GET /{version}/continents/{ccode}               |   GET /router.php?version={version}&controller=continents&method=read&ccode={ccode}             |
|   Crée un continent               |   POST /{version}/continents                      |   POST /router.php?version={version}&controller=continents&method=create                        |
|   Modifie un continent            |   POST /{version}/continents/{ccode}              |   POST /router.php?version={version}&controller=continents&method=update&ccode={ccode}          |
|   Supprime un continent           |   DELETE /{version}/continents/{ccode}            |   DELETE /router.php?version={version}&controller=continents&method=delete&ccode={ccode}        |
|   Recherche un continent par nom  |   GET /{version}/continents/search/cname={cname}  |   GET /router.php?version={version}&controller=continents&method=search&cname={cname}           |
|-----------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------------|
|   Liste les pays d'un continent   |   GET /v2/continents/{ccode}/countries            |   GET /router.php?version=v2&controller=countries&method=list&ccode={ccode}                     |
|   Accède à un pays de continent   |   GET /v2/continents/{ccode}/countries/{pcode}    |   GET /router.php?version=v2&controller=countries&method=read&ccode={ccode}&pcode={pcode}       |
|   Crée un pays de continent       |   POST /v2/continents/{ccode}/countries           |   POST /router.php?version=v2&controller=countries&method=create&ccode={ccode}                  |
|   Modifie un pays de continent    |   POST /v2/continents/{ccode}/countries/{pcode}   |   POST /router.php?version=v2&controller=countries&method=update&ccode={ccode}&pcode={pcode}    |
|   Supprime un pays de continent   |   DELETE /v2/continents/{ccode}/countries/{pcode} |   DELETE /router.php?version=v2&controller=countries&method=delete&ccode={ccode}&pcode={pcode}  |
|-----------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------------|
|   Documentation de l'API (.md)    |   GET /{version}/{controller}/info                |   GET /router.php?version={version}&controller={controller}&method=info                         |
|-----------------------------------|---------------------------------------------------|-------------------------------------------------------------------------------------------------|


## PARAMETRES DE CHEMIN OBLIGATOIRES

Les endpoints partagent trois paramètres de chemin obligatoire :

1. Numéro de version de l'API :
    - nom : `version`
    - type : énumération
    - valeurs autorisées : `v1`, `v2`

2. Le controleur en charge de l'opération :
    - nom : `controller`
    - type : énumération
    - valeurs autorisées : `continents`, `countries`

3. La méthode du controleur à invoquer :
    - nom : `method`
    - type : énumération
    - valeurs autorisées : `create,read,update,delete,list,search,info`

Le paramètre `controller` permet de différencier les 2 routes et de requêter :
    - soit des continents (valeur `continents`),
    - soit des pays (valeur `countries`).

Le paramètre `method` permet de différencier les opérations à mener pour la route empruntée.

La route des pays n'est accessible qu'en version `v2`.

La documentation est accessible avec la méthode `info`.


## CODES D'ETAT DE LA REPONSE HTTP

[Un paramètre obligatoire qui est absent ou invalide génère le code de réponse HTTP 400](https://fr.wikipedia.org/wiki/Liste_des_codes_HTTP)

La valeur `v1` pour le paramètre `version` pour la route `countries` génère le code 307 de redirection temporaire vers l'URL correspondant à la version `v2`.
# REPONSES

## TYPE DE REPONSE

Une réponse est une collection d'objets de même type : soit des continents, soit des pays.
Une collection peut contenir 0, 1 ou plusieurs objets.

Propriétés d'un objet de type `continent` :
    - `code` (string) : code à 2 lettres du continent - stocké dans `continent_codes.json`
    - `name` (string) : nom du continent

Propriétés d'un objet de type `country` :
    - `code` (string) : code du pays à 2 lettres au format [ISO 3166 Alpha 2](https://en.wikipedia.org/wiki/List_of_sovereign_states_and_dependent_territories_by_continent_(data_file)) - stocké dans `country_codes.json`
    - `name` (string) : nom du pays
    - `government` (string) : type de gouvernement - stocké dans `country_features.json`
    - `life_expectancy` (float) : espérance de vie moyenne donnée avec une précision de 1 chiffre après la virgule - stocké dans `country_features.json`
    - `continent_code` (string) : code du continent du pays - accessible via `country_continents.json` et `continent_codes.json`


## FORMAT DE REPONSE

Le format de la réponse est dicté par un paramètre de chemin optionnel que partagent tous les endpoints :
    - nom : format
    - type : énumération
    - valeurs autorisées : `csv`, `json`

S'il est communiqué, l'invalidité de ce paramètre génère le code de réponse HTTP 400.

Selon sa valeur, le type [MIME]((https://fr.wikipedia.org/wiki/Multipurpose_Internet_Mail_Extensions)) de la réponse figure dans l'en-tête Content-type de la réponse HTTP :
    - `csv` : text/csv
    - `json` : application/json

Si le paramètre est non communiqué, le format `json` s'applique par défaut.# ENDPOINTS CONTINENTS
  
## LISTE DES CONTINENTS 

- Renvoie la liste des continents
- GET /{version}/continents
- Exemple : GET version=v2&controller=continents&method=list
- Réponse :
    - csv : code;nom\nAF;Africa\nAS;Asia\n ...
    - json : [{"code: "AF", "nom" : "Africa"},{"code": "AS", "nom" : "Asia"}, ...]

## ACCES A UN CONTINENT

- Renvoie le continent correspondant au code demandé.
- GET /{version}/continents/{ccode}
- Paramètres de chemin :
    - `ccode` (string) : code du continent demandé
- Exemple
    - GET version=v2&controller=continents&method=read&ccode=EU
    - Réponse selon format :
        - csv : code;nom\nEU;Europe
        - json : [{"code" : "EU", "nom" : "Europe"}]

## CREATION D'UN CONTINENT

- Crée un continent.
- POST /{version}/continents
- Paramètres du corps de requête :
    - `ccode` (string) : code du continent à créer en 2 lettres majuscules
    - `name` (string) : nom du continent à créer
- Exemple
    - POST version=v2&controller=continents&method=create avec ccode=AT et name=Atlantide
    - Réponse selon format :
        - csv : code;nom\nAT;Atlantide
        - json : [{"code" : "AT", "nom" : "Atlantide"}]

## MODIFICATION D'UN CONTINENT

- Modifie le nom du continent correspondant au code demandé.
- POST /{version}/continents/{ccode}
- Paramètres de chemin :
    - `ccode` (string) : code du continent à modifier en 2 lettres majuscules
- Paramètres du corps de requête :
    - `name` (string) : nouveau nom du continent
- Exemple
    - POST version=v2&controller=continents&method=update&ccode=EU et name=Eden
    - Réponse selon format :
        - csv : code;nom\nEU;Eden
        - json : [{"code" : "EU", "nom" : "Eden"}]

## SUPPRESSION D'UN CONTINENT

- Supprime le continent correspondant au code demandé.
- DELETE /{version}/continents/{ccode}
- Paramètres de chemin :
    - `ccode` (string) : code du continent à supprimer en 2 lettres majuscules
- Exemple
    - DELETE version=v2&controller=continents&method=update&ccode=EU
    - Réponse selon format :
        - csv : code;nom\nEU;Europe
        - json : [{"code" : "EU", "nom" : "Europe"}]

## RECHERCHE D'UN CONTINENT PAR NOM

- Recherche le continent correspondant au nom demandé.
- GET /{version}/continents/search/{cname}
- Paramètres de chemin :
    - `cname` (string) : nom du continent à rechercher
- Exemple
    - GET version=v2&controller=continents&method=search&cname=Europe
    - Réponse selon format :
        - csv : code;nom\nEU;Europe
        - json : [{"code" : "EU", "nom" : "Europe"}]
