let anneeBissextile = function annee -> 
  if annee mod 4 = 0 then
    if annee mod 100 = 0 then
      if annee mod 400 = 0 then
        true
      else
        false
    else
      true
  else
    false ;;

anneeBissextile 2000 ;;