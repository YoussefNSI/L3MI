BEGIN {
    FS = " "
    nbFichiers = 0
    taille = 0
}
{
    nbFichiers++
    if(NF >= 5 && $5 != "")
        taille += $5
}
END {
    print "Il y a " nbFichiers " fichiers pour un total de " taille " octets."
}