%{
#include "scanner.hh"

#define YY_NO_UNISTD_H

%}

%option c++
%option yyclass="Scanner"
%option noyywrap

%%

[a-zA-Z]+   {
    nbMots++;
}

[0-9]+ {
    nbNombres++;
}

[[:punct:]]  {
    nbPonctuations++;
}

.       {
    std::cout << YYText();
}

%%

int main( int argc, char* argv[] ) 
{
    Scanner* lexer = new Scanner(std::cin, std::cout);
    while(lexer->yylex() != 0);
    
    // placer son code ici pour effectuer des actions après le parsing du fichier
    lexer->affichage();

    return 0;
}
