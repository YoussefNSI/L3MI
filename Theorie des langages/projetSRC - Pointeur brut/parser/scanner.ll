%{

#include "scanner.hh"
#include <cstdlib>
#include <iostream>
#include <locale.h>

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

[ \t]+         ; 
[\r\n]+         { loc->lines(yyleng); loc->step(); }

!T\[[0-9]+\] {
    int indice = atoi(yytext + 2);
    yylval->emplace<int>(indice);
    std::cout << "[SCAN] TITRE indice : " << indice << std::endl;
    return token::TITRE_INDICE;
}

!P\[[0-9]+\] {
    int indice = atoi(yytext + 2);
    yylval->emplace<int>(indice);
    std::cout << "[SCAN] PARAGRAPHE indice : " << indice << std::endl;
    return token::PARAGRAPHE_INDICE;
}

!I\[[0-9]+\] {
    int indice = atoi(yytext + 2);
    yylval->emplace<int>(indice);
    std::cout << "[SCAN] IMAGE indice : " << indice << std::endl;
    return token::IMAGE_INDICE;
}

!T+ {
    int length = yyleng - 1;
    if (length == 1) {
        yylval->emplace<TitreInfo>(TitreInfo{length});
        std::cout << "[SCAN] TITRE niveau : " << yylval->as<TitreInfo>().niveau << std::endl;
        return token::TITRE;
    } else if (length >= 2) {
        yylval->emplace<TitreInfo>(TitreInfo{length});
        std::cout << "[SCAN] SOUS_TITRE niveau : " << yylval->as<TitreInfo>().niveau << std::endl;
        return token::SOUS_TITRE;
    }
}

!P            { std::cout << "[SCAN] PARAGRAPHE" << std::endl; return token::PARAGRAPHE;
                return token::PARAGRAPHE; }

!I            {  std::cout << "[SCAN] IMAGE" << std::endl;
                return token::IMAGE;}

@DEFINE       { std::cout << "[SCAN] DEFINE" << std::endl;
                return token::DEFINE; }
@TITREPAGE    { std::cout << "[SCAN] TITREPAGE" << std::endl;
                return token::TITREPAGE; }
@STYLE        { std::cout << "[SCAN] STYLE" << std::endl;
                return token::STYLE; }



page            { std::cout << "[SCAN] PAGE" << std::endl;
                yylval->emplace<std::string>(std::string("page"));
                return token::BLOCS; }
titre[1-9]        { std::cout << "[SCAN] TITRE (style)" << std::endl;
                yylval->emplace<std::string>(std::string(yytext));
                return token::BLOCS; }
paragraphe      { std::cout << "[SCAN] PARAGRAPHE (style)" << std::endl;
                yylval->emplace<std::string>(std::string("paragraphe"));
                return token::BLOCS; }

rgb\(([0-9]+),([0-9]+),([0-9]+)\) {
    int r, g, b;
    sscanf(yytext, "rgb(%d,%d,%d)", &r, &g, &b);
    char hexColor[8];
    snprintf(hexColor, sizeof(hexColor), "#%02X%02X%02X", r, g, b);
    yylval->emplace<std::string>(std::string(hexColor));
    std::cout << "[SCAN] RGB : " << yytext << std::endl;
    return token::RGB_COULEUR;
}

#[0-9a-fA-F]{6} {
    yylval->emplace<std::string>(yytext);
    std::cout << "[SCAN] HEX : " << yytext << std::endl;
    return token::HEX_COULEUR;
    }

px      ;

couleurTexte    { std::cout << "[SCAN] COULEURTEXTE" << std::endl; return token::COULEURTEXTE; }
couleurFond     { std::cout << "[SCAN] COULEURFOND" << std::endl; return token::COULEURFOND; }
opacite         { std::cout << "[SCAN] OPACITE" << std::endl; return token::OPACITE; }
largeur         { std::cout << "[SCAN] LARGEUR" << std::endl; return token::LARGEUR; }
hauteur         { std::cout << "[SCAN] HAUTEUR" << std::endl; return token::HAUTEUR; }
style           { std::cout << "[SCAN] STYLE" << std::endl; return token::SELECTSTYLE; }

encodage      {
    std::cout << "[SCAN] ENCODAGE" << std::endl;
    yylval->emplace<std::string>(std::string(yytext));
    return token::PROPRIETE;
}
icone        {
    std::cout << "[SCAN] ICONE" << std::endl;
    yylval->emplace<std::string>(std::string(yytext));
    return token::PROPRIETE;
}
css          {
    std::cout << "[SCAN] CSS" << std::endl;
    yylval->emplace<std::string>(std::string(yytext));
    return token::PROPRIETE;
}
langue       {
    std::cout << "[SCAN] LANGUE" << std::endl;
    yylval->emplace<std::string>(std::string(yytext));
    return token::PROPRIETE;
}

\%\%[^\n]*[\n]         {
    std::string comment(yytext + 2);
    yylval->emplace<std::string>(comment);
    std::cout << "[SCAN] COMMENTAIRE : " << comment << std::endl;
    return token::COMMENTAIRE;
}; 
\%\%\%(.|\n)*?\%\%\%    {
    std::string comment(yytext + 3, yyleng - 6);
    std::cout << "[SCAN] COMMENTAIRE : " << comment << std::endl;
    yylval->emplace<std::string>(comment);
    return token::COMMENTAIRE;
}; 

SI            { std::cout << "[SCAN] SI" << std::endl; return token::SI; }
SINON         { std::cout << "[SCAN] SINON" << std::endl; return token::SINON; }
FINSI         { std::cout << "[SCAN] FINSI" << std::endl; return token::FINSI; }
POUR          { std::cout << "[SCAN] POUR" << std::endl; return token::POUR; }
FINI          { std::cout << "[SCAN] FINI" << std::endl; return token::FINI; }

\=             { std::cout << "[SCAN] =" << std::endl; return token::EGAL; }
\[             { std::cout << "[SCAN] [" << std::endl; return token::CROCHET_OUVRANT; }
\]             { std::cout << "[SCAN] ]" << std::endl; return token::CROCHET_FERMANT; }
\,             { std::cout << "[SCAN] ," << std::endl; return token::VIRGULE; }
\:             { std::cout << "[SCAN] :" << std::endl; return token::DEUX_POINTS; }
\(             { std::cout << "[SCAN] (" << std::endl; return token::PARENTHESE_OUVRANTE; }
\)             { std::cout << "[SCAN] )" << std::endl; return token::PARENTHESE_FERMANTE; }
\{             { std::cout << "[SCAN] {" << std::endl; return token::ACCOLADE_OUVRANTE; }
\}             { std::cout << "[SCAN] }" << std::endl; return token::ACCOLADE_FERMANTE; }
\;             { std::cout << "[SCAN] ;" << std::endl; return token::POINT_VIRGULE; }
\.             { std::cout << "[SCAN] ." << std::endl; return token::POINT; }
\+             { std::cout << "[SCAN] +" << std::endl; return token::PLUS; }
\-             { std::cout << "[SCAN] -" << std::endl; return token::MOINS; }
\*             { std::cout << "[SCAN] *" << std::endl; return token::MULT; }
\/             { std::cout << "[SCAN] /" << std::endl; return token::DIV; }

[0-9]+          { std::cout << "[SCAN] ENTIER : " << yytext << std::endl;
    yylval->emplace<int>(atoi(yytext)); return token::ENTIER; }
    
[a-zA-Z_][a-zA-Z0-9_]*   {  std::cout << "[SCAN] IDENTIFIANT : " << yytext << std::endl;
                            yylval->emplace<std::string>(yytext); return token::IDENTIFIANT; }

\'[^']*\'       {   std::string str(yytext+1, yyleng-2); 
                std::cout << "[SCAN] CHAINE : " << str << std::endl;
                yylval->emplace<std::string>(str); return token::CHAINE; }

.               {
    std::cerr << "Erreur: Caractère invalide '" << yytext[0] << "' ligne " 
              << loc->begin.line << ", colonne " << (loc->begin.column - 1) << std::endl;
              };

%%
