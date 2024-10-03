
let rec longueur = function li ->
                             match li with
                             [] -> 0
                             |_::r -> 1 + longueur r;;

let concat = function l1 -> function l2 ->
                              l1 @ l2;;

let rec nieme = function li -> function n ->
                                         match (li, n) with
                                         |([],_) -> failwith "Element introuvable ou liste vide"
                                         |(x::r,1) -> x
                                         |(x::r,_) -> nieme r (n-1);;
longueur [1;5;3];;
concat [1;2] [3;4];;
nieme [1;2;3;4;5] 4;;
