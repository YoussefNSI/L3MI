let g1 =
  [(1, 0, [2;4]) ; (2, 0, [1;3]) ; (3, 0, [2;5]) ; (4, 0, [1;5;6]) ;
   (5, 0, [3;4;9;10]) ; (6, 0, [4;7;8]) ; (7, 0, [6]) ;
   (8, 0, [6;11;12]) ; (9, 0, [5;13]) ; (10, 0, [5;11]) ;
   (11, 0, [8;10]) ; (12, 0, [8;16]) ; (13, 0, [9;14;15]) ;
   (14, 0, [13;15;17]) ; (15, 0, [13;14]) ; (16, 0, [12;17]) ;
   (17, 0, [14;16;18;19]) ; (18, 0, [17;19]) ; (19, 0, [17;18]) ];;

let rec successeurs sommet graphe =
  match graphe with
  | [] -> []
  | (s, _, l)::q when s = sommet -> l
  | _::q -> successeurs sommet q;;

let rec est_marque sommet graphe =
  match graphe with
  | [] -> false
  | (s, 0, _)::q when s = sommet -> false
  | (s, m, _)::q when s = sommet && m > 0 -> true
  | _::q -> est_marque sommet q;;

let succ_marques sommet graphe =
  try
    let (_, _, successeurs) = List.find (fun (s, _, _) -> s = sommet) graphe in
    List.filter (fun x -> est_marque x graphe) successeurs
  with Not_found -> [];;
  

succ_marques 1 g1;;
est_marque 1 g1;;
successeurs 1 g1;;

let rec marquer nbr sommet graphe =
  match graphe with
  | [] -> []
  | (s, m, l)::q when s = sommet -> (s, m + nbr, l) :: q
  | t::q -> t :: marquer nbr sommet q;;

marquer 3 1 g1;;

let rec tous_chemins sommet_init sommet_fin longueur graphe =
  if longueur = 0 then
    if sommet_init = sommet_fin then [[sommet_fin]] else []
  else
    let succs = successeurs sommet_init graphe in
    List.fold_left (fun chemins_acc s ->
      if s = sommet_init then
      chemins_acc
      else
      let chemins = tous_chemins s sommet_fin (longueur - 1) graphe in
      let chemins = List.map (fun chemin -> sommet_init :: chemin) chemins in
      chemins @ chemins_acc
    ) [] succs;;

tous_chemins 5 8 3 g1;;

let rec marquer_chemin l_sommets graphe =
  match l_sommets with
  | [] -> graphe
  | s::q -> marquer 1 s (marquer_chemin q graphe);;

marquer_chemin [1;2;3] g1;;

let rec parcours_retour sommet_actuel sommet_fin graphe_marque =
  if sommet_actuel = sommet_fin then
    [sommet_fin]
  else if not (est_marque sommet_actuel graphe_marque) then
    []  (* Si le sommet actuel n'est pas marqué, on ne peut pas continuer *)
  else
    let succs = succ_marques sommet_actuel graphe_marque in
    let rec aux = function
      | [] -> []  (* Aucun chemin trouvé dans cette branche *)
      | s :: q ->
          if not (est_marque s graphe_marque) then
            aux q
          else
            let chemin = parcours_retour s sommet_fin graphe_marque in
            if chemin = [] then aux q
            else sommet_actuel :: chemin  (* Chemin trouvé, on le retourne *)
    in
    match aux succs with
    | [] -> []  (* Aucun chemin trouvé *)
    | chemin -> chemin;;

parcours_retour 5 1 (marquer_chemin [1;2;3;5;4] g1);;

let generateur = Random.self_init ();;
(* Fonction pour enlever aléatoirement des marques dans le graphe *)
let enlever_marques n chemin graphe =
  (* Fonction auxiliaire qui enlève la marque d’un sommet donné dans le graphe *)
  let rec enlever_marque sommet graphe =
    match graphe with
    | [] -> []
    | (s, m, successeurs) :: q ->
        if s = sommet then
          (s, 0, successeurs) :: q  (* Enlève la marque en mettant `m = 0` *)
        else
          (s, m, successeurs) :: enlever_marque sommet q
  in
  (* Parcourt le chemin et enlève des marques selon le pourcentage `n` *)
  let rec aux premier sommet_liste graphe =
    match sommet_liste with
    | [] -> graphe  (* Si la liste est vide, retourner le graphe tel quel *)
    | sommet :: q when sommet = premier -> aux premier q graphe
    | sommet :: q ->
        (* On décide aléatoirement de retirer la marque du sommet *)
        let graphe =
          if Random.int 100 < n then
            enlever_marque sommet graphe
          else
            graphe
        in
        aux premier q graphe
  in
  match chemin with
  | [] -> graphe
  | premier :: reste -> aux premier reste graphe;;

enlever_marques 100 [1;2;3;5;4] (marquer_chemin [1;2;3;5;4] g1);;

let cherche_marque distance_min distance_max sommet_initial graphe =
  let rec bfs file visitees distance =
    match file with
    | [] -> None  (* Si la file est vide, aucune marque trouvée *)
    | (courant, chemin) :: reste ->
        if List.mem courant visitees then
          bfs reste visitees distance  (* Ignorer les sommets déjà visités *)
        else if distance > distance_max then
          None  (* On a dépassé la distance maximale, arrêter *)
        else if distance >= distance_min && est_marque courant graphe then
          Some (courant, List.rev chemin)  (* Sommet marqué trouvé dans la bonne plage de distance *)
        else
          (* Continuer à explorer les voisins du sommet actuel *)
          let voisins = successeurs courant graphe in
          let nouvelle_file = reste @ List.map (fun s -> (s, s :: chemin)) voisins in
          bfs nouvelle_file (courant :: visitees) (distance + 1)
  in
  bfs [(sommet_initial, [sommet_initial])] [] 0;;


let g1_marq3 = marquer_chemin [1; 4; 5; 10; 11; 8; 12; 16; 17; 18] g1;;
let g1_marq3_bis = enlever_marques 50 [1; 4; 5; 10; 11; 8; 12; 16; 17; 18] g1_marq3;;

let g1_marq3_bis = [(1, 1, [2; 4]); (2, 0, [1; 3]); (3, 0, [2; 5]); (4, 1, [1; 5; 6]);
 (5, 0, [3; 4; 9; 10]); (6, 0, [4; 7; 8]); (7, 0, [6]);
 (8, 1, [6; 11; 12]); (9, 0, [5; 13]); (10, 0, [5; 11]); (11, 1, [8;10]);
 (12, 0, [8; 16]); (13, 0, [9; 14; 15]); (14, 0, [13; 15; 17]);
 (15, 0, [13; 14]); (16, 1, [12; 17]); (17, 0, [14; 16; 18; 19]);
 (18, 1, [17; 19]); (19, 0, [17; 18])];;

cherche_marque 4 4 9 g1_marq3_bis;;


