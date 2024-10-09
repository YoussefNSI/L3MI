#!/bin/bash


for x in $(ls);
do
    if [ $(file -i $x | cut -d'=' -f2) == 'iso-8859-1' ];
    then
        $(iconv -f Latin1 -t utf-8 $x > utf8_$x)
    fi
done