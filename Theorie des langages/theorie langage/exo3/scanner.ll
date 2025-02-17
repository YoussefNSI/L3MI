%{
#include "scanner.hh"

#define YY_NO_UNISTD_H

%}

%option c++
%option yyclass="Scanner"
%option noyywrap

%%

'begin'|'end'  {
    std::cout << "-> mot clé : " << YYText();
}

[0-9]+(,[0-9]+)*    {
    std::cout << "-> nombre : " << YYText();
}

[a-zA-Z]+([0-9]|[a-zA-Z])*  {
    std::cout << "-> identificateur : " << YYText();
}

(('+'|'-')|('*'|'**'))    {
    std::cout << "-> operateur : " << YYText();
}

.       {
    std::cerr << "erreur";
    YYERROR;
}



%%

int main( int argc, char* argv[] ) 
{
    Scanner* lexer = new Scanner(std::cin, std::cout);
    while(lexer->yylex() != 0);
    
    // placer son code ici pour effectuer des actions après le parsing du fichier
    return 0;
}
