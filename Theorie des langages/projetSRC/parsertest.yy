%{
#include "ast/bloc.h"
#include "driver.h"
#include <iostream>
%}

%union {
    int ival;
    std::string* sval;
    Bloc* bloc;
}

%token TITRE SOUS_TITRE PARAGRAPHE IMAGE
%token DEFINE TITREPAGE STYLE
%token COMMENTAIRE_LIGNE COMMENTAIRE_BLOC
%token ATTRIBUT
%token SI SINON FINSI POUR FINI
%token ENTIER CHAINE
%token EGAL CROCHET_OUVRANT CROCHET_FERMANT VIRGULE DEUX_POINTS

%type <bloc> bloc

%%

program:
    blocs
    ;

blocs:
    bloc
    | blocs bloc
    ;

bloc:
    TITRE CHAINE { $$ = new Titre(*$2); }
    | SOUS_TITRE CHAINE { $$ = new SousTitre(*$2); }
    | PARAGRAPHE CHAINE { $$ = new Paragraphe(*$2); }
    | IMAGE CHAINE { $$ = new Image(*$2); }
    | DEFINE CHAINE CHAINE { $$ = new Define(*$2, *$3); }
    | TITREPAGE CHAINE { $$ = new TitrePage(*$2); }
    | STYLE CHAINE CROCHET_OUVRANT attributs CROCHET_FERMANT { $$ = new Style(*$2, $4); }
    ;

attributs:
    attribut
    | attributs VIRGULE attribut
    ;

attribut:
    CHAINE DEUX_POINTS CHAINE { /* Créer un attribut */ }
    ;

%%

int main(int argc, char** argv) {
    yyparse();
    return 0;
}

void yyerror(const char* s) {
    std::cerr << "Error: " << s << std::endl;
}