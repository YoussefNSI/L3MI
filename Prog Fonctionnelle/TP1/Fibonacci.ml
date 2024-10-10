let suiteDeFibonacci = function n ->
    let rec aux = function a -> function b -> function n ->
      if n = 0 then
        a
      else
        aux b (a + b) (n - 1)
    in
    aux 0 1 n ;;

    