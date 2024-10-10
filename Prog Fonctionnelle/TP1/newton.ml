let racineCarreeNewton = function n -> 
  let rec racineCarreeAux = function x -> function n -> 
    if abs_float (x *. x -. n) < 0.0001 then
      x
    else
      racineCarreeAux ((x +. n /. x) /. 2.0) n in
  racineCarreeAux 1.0 n ;;