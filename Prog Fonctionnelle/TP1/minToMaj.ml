let minToMaj = function c -> 
  if c >= 'a' && c <= 'z' then
    char_of_int (int_of_char c - int_of_char 'a' + int_of_char 'A')
  else
    c ;;