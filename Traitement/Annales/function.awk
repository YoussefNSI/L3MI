BEGIN {
    function_count = 0
    argument_count = 0
}

/function/ {
    function_count++
    match($0, /\([^)]*\)/)
    arguments = substr($0, RSTART + 1, RLENGTH - 2)
    split(arguments, args, ",")
    argument_count += length(args)
}

END {
    print "Il y a " argument_count " arguments répartis dans " function_count " fonctions."
}