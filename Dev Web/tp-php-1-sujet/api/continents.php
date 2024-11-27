<?php
function getContinents() {
    $file = __DIR__ . "/data/continent_codes.json";
    $file_contents = file_get_contents($file);
    return json_decode($file_contents, true);
}

function checkContinentCode($ccode) {
    return (in_array($ccode, array_keys(getContinents())));
}

function checkContinentName($cname) {
    return (in_array($cname, array_values(getContinents())));
}

// get fields from path or body or else set to false
$ccode = $_GET["ccode"] ?? ($_POST["ccode"] ?? false);
$cname = $_GET["cname"] ?? ($_POST["cname"] ?? false);

// checking endpoint consistency
if(     ($controller === "continents" && $method === "list" && ($ccode || $cname))
    ||  ($controller === "continents" && $method === "read" && (! checkContinentCode(($_GET["ccode"] ?? false)) || $cname))
    ||  ($controller === "continents" && $method === "create" && (!($_POST["ccode"] ?? false) || !($_POST["cname"] ?? false)))
    ||  ($controller === "continents" && $method === "update" && (! checkContinentCode(($_GET["ccode"] ?? false)) || !($_POST["cname"] ?? false)))
    ||  ($controller === "continents" && $method === "delete" && (! checkContinentCode(($_GET["ccode"] ?? false)) || $cname))
    ||  ($controller === "continents" && $method === "search" && ($ccode || ! checkContinentName($_GET["cname"]))))
    {
        $continents = getContinents();
        var_dump($_GET);
        var_dump($_POST);
        echo "Invalid request parameters ($ccode,$cname) for $method:
        \nContinent codes :" . array_walk($continents, function($v,$k) { echo $k, ", "; })
        . "\nContinent names :" . array_walk($continents, function($v) { echo $v, ", "; } );
        http_response_code(400);
        exit();
    }

// converts any array $tab of continents [["EU" => "Europe"], ...] into [["code" => "EU", "nom" => "Europe"], ...]
function rekey($tab) {
    return
    array_values( // re-indexes array numerically
        array_map(function($v) use($tab) { // 2nd parameter of callback is NOT the current element's key with array_map (as opposed to other array_* functions ...) 
            return ["code"=>array_search($v, $tab), "nom"=>$v]; // so search key with array_search passing array with `use`
        },
        $tab)
    );
}

$str .=<<<DOC

# ENDPOINTS CONTINENTS
  
## LISTE DES CONTINENTS 

- Renvoie la liste des continents
- GET /{version}/continents
- Exemple : GET version=v2&controller=continents&method=list
- Réponse :
    - csv : code;nom\\nAF;Africa\\nAS;Asia\\n ...
    - json : [{"code: "AF", "nom" : "Africa"},{"code": "AS", "nom" : "Asia"}, ...]

DOC;


if($controller === "continents" && $method === "list") {
    $continents = getContinents();
    ksort($continents);
    $response = rekey($continents);
    echo format($response);
    exit();
}

$str .=<<<DOC

## ACCES A UN CONTINENT

- Renvoie le continent correspondant au code demandé.
- GET /{version}/continents/{ccode}
- Paramètres de chemin :
    - `ccode` (string) : code du continent demandé
- Exemple
    - GET version=v2&controller=continents&method=read&ccode=EU
    - Réponse selon format :
        - csv : code;nom\\nEU;Europe
        - json : [{"code" : "EU", "nom" : "Europe"}]

DOC;

if($controller === "continents" && $method === "read") {
    $continents = getContinents();
    if(! key_exists($ccode, $continents)) {
        echo "Continent access: continent code does not exist";
        http_response_code(400);
        exit();
    }
    
    $response = [$ccode => $continents[$ccode]];
    $response = rekey($response);
    echo format($response);
    exit();
}

$str .=<<<DOC

## CREATION D'UN CONTINENT

- Crée un continent.
- POST /{version}/continents
- Paramètres du corps de requête :
    - `ccode` (string) : code du continent à créer en 2 lettres majuscules
    - `name` (string) : nom du continent à créer
- Exemple
    - POST version=v2&controller=continents&method=create avec ccode=AT et name=Atlantide
    - Réponse selon format :
        - csv : code;nom\\nAT;Atlantide
        - json : [{"code" : "AT", "nom" : "Atlantide"}]

DOC;
    
if($controller === "continents" && $method === "create") {
    $continents = getContinents();
    if(key_exists($ccode, $continents)) {
        echo "Continent creation: continent code already exists";
        http_response_code(400);
        exit();
    }

    $continents[$ccode] = $cname;
    $file = __DIR__ . "/data/continent_codes.json";
    file_put_contents($file, json_encode($continents));
    $response = [$ccode => $cname];
    $response = rekey($response);
    echo format($response);
    exit();
}

$str .=<<<DOC

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
        - csv : code;nom\\nEU;Eden
        - json : [{"code" : "EU", "nom" : "Eden"}]

DOC;

if($controller === "continents" && $method === "update") {
    $continents = getContinents();
    if(! key_exists($ccode, $continents)) {
        echo "Continent modification: continent code does not exist";
        http_response_code(400);
        exit();
    }

    $continents[$ccode] = $cname;
    $file = __DIR__ . "/data/continent_codes.json";
    file_put_contents($file, json_encode($continents));
    $response = [$ccode => $continents[$ccode]];
    $response = rekey($response);
    echo format($response);
    exit();
}

$str .=<<<DOC

## SUPPRESSION D'UN CONTINENT

- Supprime le continent correspondant au code demandé.
- DELETE /{version}/continents/{ccode}
- Paramètres de chemin :
    - `ccode` (string) : code du continent à supprimer en 2 lettres majuscules
- Exemple
    - DELETE version=v2&controller=continents&method=update&ccode=EU
    - Réponse selon format :
        - csv : code;nom\\nEU;Europe
        - json : [{"code" : "EU", "nom" : "Europe"}]

DOC;

if($controller === "continents" && $method === "delete") {
    $continents = getContinents();
    if(! key_exists($ccode, $continents)) {
        echo "Continent removal: continent code does not exist";
        http_response_code(400);
        exit();
    }

    $response = [$ccode => $continents[$ccode]];
    $response = rekey($response);
    unset($continents[$ccode]);
    $file = __DIR__ . "/data/continent_codes.json";
    file_put_contents($file, json_encode($continents));
    echo format($response);
    exit();
}

$str .=<<<DOC

## RECHERCHE D'UN CONTINENT PAR NOM

- Recherche le continent correspondant au nom demandé.
- GET /{version}/continents/search/{cname}
- Paramètres de chemin :
    - `cname` (string) : nom du continent à rechercher
- Exemple
    - GET version=v2&controller=continents&method=search&cname=Europe
    - Réponse selon format :
        - csv : code;nom\\nEU;Europe
        - json : [{"code" : "EU", "nom" : "Europe"}]

DOC;

if($controller === "continents" && $method === "search") {
    $continents = getContinents();
    $ccode = array_search($cname, $continents);
    if($ccode)
        $response = [$ccode => $cname];
    else
        $response = [];

    $response = rekey($response);
    echo format($response);
    exit();
}
?>