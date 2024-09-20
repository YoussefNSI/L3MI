#!/bin/bash

FichierEntree="";
FicAFusion=();
estEntree=true;

for x in $*;
do
    if [[ $estEntree = true ]] 
    then
        FichierEntree=$x;
        estEntree=false;
    else
        if [ -f $x ]
        then
            ENCODAGE=$(file -i $x | cut -d'=' -f2)
            if [[ $ENCODAGE == "utf-8" ]]
            then
                FicAFusion+=($x)
            elif [[ $ENCODAGE == "iso-8859-1" ]]
            then
                if [ ! -f 'utf8_'$x ]
                then
                    $(iconv -f Latin1 -t UTF8 -c $x > utf8_$x)
                fi
                FicAFusion+=("utf8_"$x);
            else
                echo "[FATAL] Encodage $ENCODAGE non supporté pour le fichier $x";
                exit 1;
            fi
        else
            echo "Le fichier $x n'existe pas";
            exit 1;
        fi
    fi
done

estEntete=true;

for x in ${FicAFusion[*]};
do
    if [[ $estEntete = true ]] 
    then
        head -n 1 $x > $FichierEntree;
        estEntete=false;
    else
        tail -n +2 $x >> $FichierEntree;
    fi
done

echo "Fusion terminée";
