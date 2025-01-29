BEGIN {
    function_count = 0  # Compteur pour le nombre de fonctions
    argument_count = 0  # Compteur pour le nombre total d'arguments
}

# Détecter les lignes contenant des définitions de fonctions
/function/ {
    function_count++  # Incrémenter le compteur de fonctions

    # Extraire les arguments entre parenthèses
    match($0, /\([^)]*\)/)
    arguments = substr($0, RSTART + 1, RLENGTH - 2)  # Extraire le contenu entre parenthèses

    # Compter le nombre d'arguments
    if (arguments != "") {
        split(arguments, args, ",")  # Diviser les arguments par des virgules
        argument_count += length(args)  # Ajouter le nombre d'arguments au compteur
    }
}

END {
    # Afficher le résultat
    print "Il y a " argument_count " arguments répartis dans " function_count " fonctions."
}