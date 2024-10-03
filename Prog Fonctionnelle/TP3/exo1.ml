type 'a arbre_binaire =
  | Feuille of 'a
  | Noeud of 'a arbre_binaire * 'a arbre_binaire;;

let arbre_binaire = Noeud(Noeud(Feuille(1), Feuille(2)), Noeud(Feuille(3), Feuille(4)));;

let rec nbNoeudsInternes = function arbre -> 
  match arbre with
    Feuille(_) -> 0
  | Noeud(Feuille(_), Feuille(_)) -> 0
  | Noeud(g, d) -> 1 + nbNoeudsInternes g + nbNoeudsInternes d;;

nbNoeudsInternes arbre_binaire;;

let rec nbFeuilles = function arbre ->
  match arbre with
    Feuille(_) -> 1
  | Noeud(g, d) -> nbFeuilles g + nbFeuilles d;;

nbFeuilles arbre_binaire;;

let rec profondeur = function arbre ->
  match arbre with
    Feuille(_) -> 0
  | Noeud(g, d) -> 1 + max (profondeur g) (profondeur d);;

profondeur arbre_binaire;;

let rec memeForme arbre1 arbre2 = 
  match arbre1, arbre2 with
    Feuille(_), Feuille(_) -> true
  | Noeud(g1, d1), Noeud(g2, d2) -> memeForme g1 g2 && memeForme d1 d2
  | _, _ -> false;;

let arbre_binaire2 = Noeud(Noeud(Noeud(Feuille(1), Feuille(2)), Feuille(2)), Noeud(Feuille(3), Feuille(4)));;

memeForme arbre_binaire arbre_binaire2;;

let rec listeFeuilles = function arbre ->
  match arbre with
    Feuille(x) -> [x]
  | Noeud(g, d) -> listeFeuilles g @ listeFeuilles d;;

listeFeuilles arbre_binaire;;

let rec mapArbre f = function arbre ->
  match arbre with
    Feuille(x) -> Feuille(f x)
    | Noeud(g, d) -> Noeud(mapArbre f g, mapArbre f d);;

let arbre_binaire3 = mapArbre (fun x -> x + 1) arbre_binaire;;

listeFeuilles arbre_binaire3;;

(* Exercice 2 *)

type operateur_bin = Mult | Add;;
type operateur_un = Moins;;
type arbre = Const of int
            | Var of string
            | Noeud1 of (operateur_un * arbre)
            | Noeud2 of (operateur_bin * arbre * arbre);;

let rec chaine_de_arbre e =
  match e with
    Const(x) -> string_of_int x
  | Var(x) -> x
  | Noeud1(Moins, a) -> "-" ^ chaine_de_arbre a
  | Noeud2(Mult, a, b) -> "(" ^ chaine_de_arbre a ^ " * " ^ chaine_de_arbre b ^ ")"
  | Noeud2(Add, a, b) -> "(" ^ chaine_de_arbre a ^ " + " ^ chaine_de_arbre b ^ ")";;

let expression = Noeud2(Add, Noeud2(Mult, Const(2), Const(6)), Const(3));;

chaine_de_arbre expression;;

let rec evaluer e =
  match e with
    Const(x) -> x
  | Var(x) -> failwith "Impossible d'évaluer une variable"
  | Noeud1(Moins, a) -> - evaluer a
  | Noeud2(Mult, a, b) -> evaluer a * evaluer b
  | Noeud2(Add, a, b) -> evaluer a + evaluer b;;

evaluer expression;;

let rec close e li = 
  match e with
    Const(x) -> Const(x)
  | Var(x) -> if List.mem_assoc x li then Const(List.assoc x li) else Var(x)
  | Noeud1(Moins, a) -> Noeud1(Moins, close a li)
  | Noeud2(Mult, a, b) -> Noeud2(Mult, close a li, close b li)
  | Noeud2(Add, a, b) -> Noeud2(Add, close a li, close b li);;

(* Exercice 3 *)

type operateur = Mult | Plus | Moins;;
type arbre = C of int | N of (operateur * arbre list);;

let rec nbConst = function arbre ->
  match arbre with
    C(_) -> 1
  | N(_, l) -> List.fold_left (fun acc x -> acc + nbConst x) 0 l;;

let arbre = N(Mult, [C(1); N(Plus, [C(2); C(3)]); C(4); N(Moins, [C(5); C(6)])]);;

nbConst arbre;;

let rec expressionCorrecte = function arbre ->
  match arbre with
    C(_) -> true
  | N(_, l) -> List.length l >= 2 && List.for_all expressionCorrecte l;; (*Equivalent a map*)

expressionCorrecte arbre;;

let rec evaluer = function arbre ->
  match arbre with
    C(x) -> x
  | N(Mult, l) -> List.fold_left (fun acc x -> acc * evaluer x) 1 l
  | N(Plus, l) -> List.fold_left (fun acc x -> acc + evaluer x) 0 l
  | N(Moins, l) -> List.fold_left (fun acc x -> acc - evaluer x) 0 l;;

evaluer arbre;;

let rec chaine_de_arbre = function arbre ->
  match arbre with
    C(x) -> string_of_int x
  | N(Mult, l) -> "(" ^ String.concat " * " (List.map chaine_de_arbre l) ^ ")"
  | N(Plus, l) -> "(" ^ String.concat " + " (List.map chaine_de_arbre l) ^ ")"
  | N(Moins, l) -> "(" ^ String.concat " - " (List.map chaine_de_arbre l) ^ ")";;

chaine_de_arbre arbre;;
