BEGIN {
    FS = ","
}
{
    malFormate = false
    if($3 != "Date de Naissance")
        if (match($3, /^[0-9]{2}\/[0-9]{2}\/[0-9]{4}$/) == 0) {
            malFormate = true
        }
        else
            print $1 " " $2 " " $3 >> "bienFormate.txt"
    else
        print $1 " " $2 " " $3 > "bienFormate.txt"
}