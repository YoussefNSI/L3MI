;;
let rec npremiers = function li -> function n ->
                                         if n < 0 then failwith "n négatif";
                                         match (li, n) with
                                         |([],_) -> failwith "n plus grand que la taille de la liste";
                                         |(x::r,1) -> x;
                                         |(x::r,_) -> x @ npremiers r (n-1);;

npremiers [1;2;3;4;5] 3;;


let rec met_a_plat = function li ->
                           match li with
                           |[] -> []
                           |x::r -> x @ (met_a_plat r);;

met_a_plat [[3;2];[5;8]];;
;

let rec paire_vers_liste = function (ln,ls) ->
    			 match (ln,ls) with
			 |([],[]) -> []
			 |(x::a,y::b) -> [(x,y)] @ paire_vers_liste (a,b);
			 | _ -> failwith "les deux listes doivent être de même longueur";;

paire_vers_liste ([1;2;3], ['a';'b';'c']);;

let rec liste_vers_paire = function ls ->
    			match ls with
			[] -> []
			|(a,b)::r ->  # a faire

list = [(1,'a'); (2,'b'); (3,'c')];
liste_vers_paire list;;

let rec supprime1 = function ls -> function e ->
    		    match ls with
		    [] -> []
		    |x::r -> if x = e
		    	     then [] @ r
			     else x :: supprime1 r e;;

supprime1 [5;3;7;9] 7;;

let rec supprime2 = function ls -> function e ->
    		    match ls with
		    [] -> []
		    |x::r -> if x = e
		    	     then [] @ supprime2 r e
			     else x :: supprime2 r e;;

supprime2 [5;3;7;11;9;7;3;7] 7;;

let min_liste = function ls ->
			  match ls with
		  |[] -> failwith "liste vide"
		  |x::r -> let rec recherche_min = function l -> function e ->
					   match l with
				   [] -> e
				   |y::r -> if y < e
						then recherche_min r y
						else recherche_min r e
				in recherche_min r x;;

min_liste [5;3;7;11;9;7;3;7];;

let rec doublon = function li ->
				match li with
				[] -> []
				|x::r -> if List.mem x r
						then doublon r
						else x :: doublon r;;

doublon [5;3;7;11;9;7;3;7];;

(* exercice 3 *)

let rec parties = function li ->
				match li with
				[] -> [[]]
				|x::r -> let rec ajoute = function l -> function e ->
							match l with
							[] -> [[e]]
							|y::r -> [e::y] @ ajoute r e
						in ajoute (parties r) x @ parties r;;

parties [1;2;3];; 

let rec sous_listes n li =
	if n = 0 then [[]]
	else match li with
			 | [] -> []
			 | x::r -> let rec ajoute = function l -> function e ->
					match l with
					[] -> [[e]]
					|y::r -> [e::y] @ ajoute r e
				in ajoute (sous_listes (n-1) r) x @ sous_listes n r;;

sous_listes 2 [1;2;3];;

let inserer_tete x ll =
	List.map (fun l -> x::l) ll;;

let rec parties2 = function li ->
				match li with
				[] -> [[]]
				|x::r -> let rec ajoute = function l -> function e ->
							match l with
							[] -> [[e]]
							|y::r -> [e::y] @ ajoute r e
						in ajoute (parties2 r) x @ inserer_tete x (parties2 r);;

parties2 [1;2;3];;

let longueur = function li ->
	List.fold_left (fun acc x -> acc + 1) 0 li;;

longueur [1;2;3;4;5];;

let met_a_plat2 = function li ->
	List.fold_left (fun acc x -> acc @ x) [] li;;

met_a_plat2 [[3;2];[5;8]];;

let supprime3 = function ls -> function e ->
	List.fold_right (fun x acc -> if x = e then acc else x::acc) ls [];;

supprime3 [5;3;7;11;9;7;3;7] 7;;

let doublon = function li ->
	List.fold_right (fun x acc -> if List.mem x acc then acc else x::acc) li [];;

doublon [5;3;7;11;9;7;3;7];;

let map2 f l1 =
	List.fold_right (fun x acc -> (f x)::acc) l1 [];;

map2 (fun x -> x+1) [1;2;3;4;5];;