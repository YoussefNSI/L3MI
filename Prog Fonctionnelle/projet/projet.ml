(*Exercice 1*)

(*Question 1*)

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
  | (s, _, l)::q when s = sommet -> l (* On retourne les successeurs du sommet *)
  | _::q -> successeurs sommet q;; (* On continue la recherche *)

let rec est_marque sommet graphe =
  match graphe with
  | [] -> false
  | (s, 0, _)::q when s = sommet -> false (* Si le sommet est marqué, on retourne faux *)
  | (s, m, _)::q when s = sommet && m > 0 -> true (* Si le sommet est marqué, on retourne vrai *)
  | _::q -> est_marque sommet q;; (* On continue la recherche *)

let succ_marques sommet graphe =
  try (* On essaie de trouver le sommet dans le graphe *)
    let (_, _, successeurs) = List.find (fun (s, _, _) -> s = sommet) graphe in (* On récupère les successeurs *)
    List.filter (fun x -> est_marque x graphe) successeurs (* On filtre les successeurs marqués pour les retourner *)
  with Not_found -> [];; (* Si le sommet n'est pas trouvé, on retourne une liste vide *)
  

succ_marques 1 g1;;
est_marque 1 g1;;
successeurs 1 g1;;

let rec marquer nbr sommet graphe =
  match graphe with
  | [] -> [] 
  | (s, m, l)::q when s = sommet -> (s, m + nbr, l) :: q (*On ajoute nbr à la marque du sommet *)
  | t::q -> t :: marquer nbr sommet q;; (* On continue la recherche *)

marquer 3 1 g1;;

(*Question 2*)

let rec tous_chemins sommet_init sommet_fin longueur graphe =
  if longueur = 0 then
    if sommet_init = sommet_fin then [[sommet_fin]] else [] 
  else
    let succs = successeurs sommet_init graphe in (* On récupère les successeurs du sommet initial *)
    List.fold_left (fun chemins_acc s -> (* On parcourt les successeurs *)
      if s = sommet_init then (* Si le successeur est le sommet initial, on ne peut pas continuer *)
      chemins_acc
      else
      let chemins = tous_chemins s sommet_fin (longueur - 1) graphe in (* On cherche les chemins de longueur - 1 *)
      let chemins = List.map (fun chemin -> sommet_init :: chemin) chemins in  (* On ajoute le sommet initial au début de chaque chemin *)
      chemins @ chemins_acc (* On ajoute les chemins trouvés à la liste des chemins *)
    ) [] succs;; 

tous_chemins 5 8 3 g1;;

let rec marquer_chemin l_sommets graphe =
  match l_sommets with 
  | [] -> graphe 
  | s::q -> marquer 1 s (marquer_chemin q graphe);; (* On marque chaque sommet de la liste *)

marquer_chemin [1;2;3] g1;;

(*Question 3*)

let rec parcours_retour sommet_actuel sommet_fin graphe_marque =
  if sommet_actuel = sommet_fin then [sommet_fin] 
  else if not (est_marque sommet_actuel graphe_marque) then []  (* Si le sommet actuel n'est pas marqué, on ne peut pas continuer *)
  else
    let succs = succ_marques sommet_actuel graphe_marque in (* On récupère les successeurs marqués *)
    let rec aux = function
      | [] -> []  (* Aucun chemin trouvé dans cette branche *)
      | s :: q ->
          if not (est_marque s graphe_marque) then
            aux q (* On ignore les sommets non marqués *)
          else
            let chemin = parcours_retour s sommet_fin graphe_marque in (* On cherche un chemin depuis le successeur *)
            if chemin = [] then aux q (* Si aucun chemin n'est trouvé, on continue la recherche *)
            else sommet_actuel :: chemin  (* Chemin trouvé, on le retourne *)
    in
    match aux succs with
    | [] -> []  (* Aucun chemin trouvé *)
    | chemin -> chemin;; 

parcours_retour 5 1 (marquer_chemin [1;2;3;5;4] g1);;

(*Avec marques manquantes Exercice 2*)

(*Question 1*)

let generateur = Random.self_init ();; (* Initialisation du générateur aléatoire *)

let enlever_marques n chemin graphe =
  let rec enlever_marque sommet graphe =
    match graphe with
    | [] -> []
    | (s, m, successeurs) :: q ->
        if s = sommet then
          (s, 0, successeurs) :: q  (* Enlève la marque*)
        else
          (s, m, successeurs) :: enlever_marque sommet q (* Continue la recherche *)
  in
  let rec aux premier sommet_liste graphe =
    match sommet_liste with
    | [] -> graphe  
    | sommet :: q when sommet = premier -> aux premier q graphe (* On ignore le premier sommet *)
    | sommet :: q ->
        let graphe =
          if Random.int 100 <= n then
            enlever_marque sommet graphe (* On enlève la marque du sommet si le nombre tiré est dans la plage 0-n% *)
          else
            graphe
        in
        aux premier q graphe
  in
  match chemin with
  | [] -> graphe (* Si le chemin est vide, on ne fait rien *)
  | premier :: reste -> aux premier reste graphe;;

enlever_marques 100 [1;2;3;5;4] (marquer_chemin [1;2;3;5;4] g1);;

(*Question 2*)


let rec cherche_marque dist_min dist_max sommet graphe =
  let rec parcours sommet dist =
    if dist > dist_max then None
    else if dist >= dist_min && est_marque sommet graphe then
      Some (sommet, [sommet])
    else
      let succs = successeurs sommet graphe in

      if dist >= dist_min && dist < dist_max then
        let succ_marque = List.find_opt (fun s -> est_marque s graphe) succs in (* Pour info find_opt est List.find mais qui retourne un "option" pour avoir un None plutot qu'une exception *)
        match succ_marque with
        | Some s -> Some (s, [sommet; s])  (* Retourne le successeur marqué trouvé *)
        | None -> 
            (* Si aucun successeur marqué trouvé, continuer la recherche normale *)
            let rec aux = function
              | [] -> None
              | s :: q ->
                  match parcours s (dist + 1) with
                  | None -> aux q
                  | Some (sommet_marque, chemin) -> Some (sommet_marque, sommet :: chemin)
            in
            aux succs
      else
        (* Si la distance est en-dessous de dist_min, continue simplement la recherche *)
        let rec aux = function
          | [] -> None
          | s :: q ->
              match parcours s (dist + 1) with
              | None -> aux q
              | Some (sommet_marque, chemin) -> Some (sommet_marque, sommet :: chemin)
        in
        aux succs
  in
  parcours sommet 0;;

let g1_marq3_bis = [(1, 1, [2; 4]); (2, 0, [1; 3]); (3, 0, [2; 5]); (4, 1, [1; 5; 6]);
 (5, 0, [3; 4; 9; 10]); (6, 0, [4; 7; 8]); (7, 0, [6]);
 (8, 1, [6; 11; 12]); (9, 0, [5; 13]); (10, 0, [5; 11]); (11, 1, [8;10]);
 (12, 0, [8; 16]); (13, 0, [9; 14; 15]); (14, 0, [13; 15; 17]);
 (15, 0, [13; 14]); (16, 1, [12; 17]); (17, 0, [14; 16; 18; 19]);
 (18, 1, [17; 19]); (19, 0, [17; 18])];; 

cherche_marque 4 4 9 g1_marq3_bis;;

(*Question 3*)

let rec parcours_retour_v2 sommet_actuel sommet_fin graphe_marque dist_max =
  if sommet_actuel = sommet_fin then [sommet_fin]
  else 
    match cherche_marque 0 dist_max sommet_actuel graphe_marque with
    | None -> [sommet_actuel; -1]
    | Some (sommet_marque, chemin_partiel) ->
        let chemin = parcours_retour_v2 sommet_marque sommet_fin graphe_marque dist_max in
        sommet_actuel :: chemin @ chemin_partiel;; (*Je suis bloqué ici, je ne sais pas si j'ai mal compris la consigne mais je ne sais pas comment procéder*)



parcours_retour_v2 18 1 g1_marq3_bis 4;;



