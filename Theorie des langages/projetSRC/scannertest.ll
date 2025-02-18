%{
#include "parser.tab.hh"
#include "ast/bloc.h"
#include "driver.h"
%}

%%

[ \t\n]+        ; // Ignore whitespace

"IT"            return TITRE;
"ITT"           return SOUS_TITRE;
"IP"            return PARAGRAPHE;
"II"            return IMAGE;

"@DEFINE"       return DEFINE;
"@TITREPAGE"    return TITREPAGE;
"@STYLE"        return STYLE;

"$$"            return COMMENTAIRE_LIGNE;
"\$\$\$"        return COMMENTAIRE_BLOC;

"!"             return ATTRIBUT;

"SI"            return SI;
"SINON"         return SINON;
"FINSI"         return FINSI;
"POUR"          return POUR;
"FINI"          return FINI;

"="             return EGAL;
"["             return CROCHET_OUVRANT;
"]"             return CROCHET_FERMANT;
","             return VIRGULE;
":"             return DEUX_POINTS;

[0-9]+          { yylval.ival = atoi(yytext); return ENTIER; }
\"[^"]*\"       { yylval.sval = new std::string(yytext); return CHAINE; }

.               return yytext[0];

%%

int yywrap() {
    return 1;
}