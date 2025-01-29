let cube x = x * x * x;;

let ascii = function c -> function n -> int_of_char c + n;;

let abs = function x -> if x < 0 then x else -x;;

let associe_nombre = function x -> 
  match x with
          | 0 -> "zero"
          | 1 -> "un"
          | 2 -> "deux"
          | 3 -> "trois"
          | 4 -> "quatre"
          | 5 -> "cinq"
          | _ -> "beaucoup";;

type genre =
  | Classique
  | Electro
  | KPop
  | Pop
  | Rap
  | RnB
  | Rock;;

type artiste = string;;

type type_album = Single | EP | Album;;

type album = artiste * type_album * string;;

type musique = string * genre * album * artiste
let meme_album = function m1 -> function m2 ->
        let (_,_,alb_a,_) = m1 in
        let (_,_,alb_b,_) = m2 in
        alb_a = alb_b;;

let meme_genre (m1 : musique) (m2 : musique) =
        let (_,genre_a,_,_) = m1 in
        let (_,genre_b,_,_) = m2 in
        genre_a = genre_b;;

let long_titre (m1 : musique) = 
    let (titre,_,_,_) = m1 in
    (String.length titre) > 20;;

type ('c, 'v) dico =
  | Vide
  | Element of 'c * 'v * ('c, 'v) dico;;

let creer = Vide;;

let rec definir (c : 'c) (v : 'v) (d : ('c, 'v) dico) = 
  match d with
  |Element(_,_,suite) -> definir c v suite
  |Vide -> Element(c, v , Vide);;

let rec obtenir (c : 'c) (d : ('c, 'v) dico) = 
  match d with
  | Vide -> failwith "Element introuvable"
  | Element(cle, v, suite) -> if c = cle then v else obtenir c suite;;

let rec supprimer (c : 'c) (d : ('c,'v) dico) =
  match d with
  | Vide -> failwith "Clé introuvable"
  | Element(cle, v, suite) -> if c = cle then suite 
                              else Element(cle, v, supprimer cle suite);;

type 'a liste =
  | Fin
  | Element of 'a * 'a liste
  
let rec associer (f : 'a -> 'b) (l : 'a liste) : 'b liste = 
  match l with
  | Fin -> Fin
  | Element(v, suite) -> Element((f v), (associer f suite));;

let rec filtrer (f : 'a -> bool) (l : 'a liste) =
  match l with 
  | Fin -> Fin
  | Element(v, suite) -> if (f v) then Element(v, filtrer f suite)
                          else filtrer f suite;;

let rec existe (f : 'a -> bool) (l : 'a liste) : bool =
  match l with
  | Fin -> false
  | Element(v, suite) -> (f v) && (existe f suite);;

let plus_grande (l1 : 'a list) (l2 : 'b list) =
   if List.length l1 < List.length l2 then l2 else l1;;

let fib (n : int) : int list =
  let rec fib_x x = 
    if x < 2 then 1
    else (fib_x (x-1)) + (fib_x (x - 2))
  in
  List.init n fib_x;;

let quarante_deux (l : int list) : int list =
  let rec ajoute_42 (x : int) = x + 42
  in
  List.map ajoute_42 l;;
  
let moyenne (l : float list) : float =
  let rec add x y = x +. y
  in
  let total = List.fold_left add 0.0 l
  in total /. (float_of_int (List.length l));;

let exercice (li : int list) : int = 
  let pair x : bool = (x mod 2) = 0 in
  let abs x = if x >= 0 then x else -x in
  let mult x y = x * y in
  let li_pair = List.filter pair li in
  let li_abs = List.map abs li_pair in
  List.fold_left mult 1 li_abs;;

type ('a, 'b) dico_binaire =
  | Fin
  | Noeud of ('a * 'b) * ('a, 'b) dico_binaire * ('a, 'b) dico_binaire;;

let rec ajouter (arb : ('a, 'b) dico_binaire) (cle : 'a) (valeur : 'b) = 
  match arb with
  | Fin -> Noeud((cle, valeur), Fin, Fin)
  | Noeud((c, v), g, d) -> if c = cle then Noeud((cle, valeur), g, d)
                           else if c < cle then 
                           Noeud((c, v), g, ajouter d cle valeur)
                           else
                           Noeud((c, v), ajouter g cle valeur, d);;

let rec trouver_nom (cle : 'a) (dic : ('a, 'b) dico_binaire) = 
  match dic with
  | Fin -> None
  | Noeud((c, v), g, d) -> 
    if c = cle then Some v
    else if cle < c then trouver_nom cle g
    else trouver_nom cle d;;

let rec supprimer (cle : 'a) (dic : ('a, 'b) dico_binaire) : ('a, 'b) dico_binaire =
  let rec trouver_min = function
    | Fin -> failwith "Arbre vide"
    | Noeud((c, v), Fin, _) -> (c, v)
    | Noeud(_, g, _) -> trouver_min g
  in
  match dic with
  | Fin -> Fin
  | Noeud((c, v), g, d) ->
    if cle < c then Noeud((c, v), supprimer cle g, d)
    else if cle > c then Noeud((c, v), g, supprimer cle d)
    else match g, d with
         | Fin, _ -> d
         | _, Fin -> g
         | _ -> let min_d = trouver_min d in
                Noeud(min_d, g, supprimer (fst min_d) d);;
                              