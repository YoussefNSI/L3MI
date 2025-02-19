%{

#include "scanner.hh"
#include <cstdlib>

#define YY_NO_UNISTD_H

using token = yy::Parser::token;

#undef  YY_DECL
#define YY_DECL int Scanner::yylex( yy::Parser::semantic_type * const lval, yy::Parser::location_type *loc )

/* update location on matching */
#define YY_USER_ACTION loc->step(); loc->columns(yyleng);

%}

%option c++
%option yyclass="Scanner"
%option noyywrap

%%
%{
    yylval = lval;
%}

[ \t\n]+        ; 

!T+ {
    int length = yyleng - 1;
    if (length == 1) {
        yylval->emplace<yy::Parser::semantic_type::TitreInfo>({yytext, 1});
        return token::TITRE;
    } else if (length >= 2) {
        yylval->emplace<yy::Parser::semantic_type::TitreInfo>({yytext, length});
        return token::SOUS_TITRE;
    }
}

!P            return token::PARAGRAPHE;

!I            return token::IMAGE;

@DEFINE       return token::DEFINE;
@TITREPAGE    return token::TITREPAGE;
@STYLE        return token::STYLE;

rgb\(([0-9]+),([0-9]+),([0-9]+)\) {
    int r, g, b;
    sscanf(yytext, "rgb(%d,%d,%d)", &r, &g, &b);
    char hexColor[8];
    snprintf(hexColor, sizeof(hexColor), "#%02X%02X%02X", r, g, b);
    yylval->emplace<std::string>(std::string(hexColor));
    return token::RGB_COULEUR;
}
#[0-9a-fA-F]{6} return token::HEX_COULEUR;

couleurTexte   return token::COULEUR_TEXTE;
couleurFond    return token::COULEUR_FOND;
opacite       return token::OPACITE;
largeur       return token::LARGEUR;
hauteur       return token::HAUTEUR;

"!"             return token::ATTRIBUT;

\%\%[^/n]*      ; // commentaire ligne
\%\%\%.*\%\%\%    ; // commentaire bloc

SI            return token::SI;
SINON         return token::SINON;
FINSI         return token::FINSI;
POUR          return token::POUR;
FINI          return token::FINI;

\=             return token::EGAL;
\[             return token::CROCHET_OUVRANT;
\]             return token::CROCHET_FERMANT;
\,             return token::VIRGULE;
\:             return token::DEUX_POINTS;
\(             return token::PARENTHESE_OUVRANTE;
\)             return token::PARENTHESE_FERMANTE;
\{             return token::ACCOLADE_OUVRANTE;
\}             return token::ACCOLADE_FERMANTE;
\;             return token::POINT_VIRGULE;
\n             return token::NEWLINE;
\r\n           return token::NEWLINE;
\r             return token::NEWLINE;

[a-zA-Z_][a-zA-Z0-9_]*   { yylval->emplace<std::string>(yytext); return token::IDENTIFIANT; }

[0-9]+          { yylval->emplace<int>(atoi(yytext)); return token::ENTIER; }
\"[^"]*\"       { yylval->emplace<std::string>(yytext); return token::CHAINE; }

.               return yytext[0];


%%
