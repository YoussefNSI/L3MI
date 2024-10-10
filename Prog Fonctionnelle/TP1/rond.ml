let rec rond = function f -> function g -> function x -> f (g x) ;;

rond (function x -> x + 1) (function x -> x * 2) 3 ;;