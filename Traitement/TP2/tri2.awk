BEGIN {
    FS = ","
}
{
    prenom = ""
    n = length($1)
    for (i = 1; i <= n; i++) {
        char = substr($1, i, 1)
        
        if (i == 1 || substr($1, i-1, 1) == " " || substr($1, i-1, 1) == "-") {
            prenom = prenom toupper(char)
        } else {
            prenom = prenom tolower(char)
        }
    }
    
    if (prenom != "" && $2 != "") {
        print prenom " " toupper($2)
    }
}
