%skeleton "lalr1.cc"
%require "3.0"

%defines
%define api.parser.class { Parser }
%define api.value.type variant
%define parse.assert


%locations

%code requires{
    #include "contexte.hh"
    #include "expressionBinaire.hh"
    #include "expressionUnaire.hh"
    #include "constante.hh"
    #include "variable.hh"
    #include "bloc.h"

    class Scanner;
    class Driver;
    class Bloc;

    struct TitreInfo{
        std::string texte;
        int niveau;
    }; 

}

%parse-param { Scanner &scanner }
%parse-param { Driver &driver }

%code{
    #include <iostream>
    #include <string>
    #include <memory>
    #include <map>
    #include <variant>
    
    #include "scanner.hh"
    #include "driver.hh"
    #include "bloc.h"


    std::unique_ptr<Document> doc = std::make_unique<Document>();


    #undef  yylex
    #define yylex scanner.yylex
}

%token                 NEWLINE
%token <TitreInfo>     TITRE SOUS_TITRE
%token <std::string*>   PARAGRAPHE IMAGE
%token <std::string*>   DEFINE TITREPAGE STYLE
%token <std::string*>   ATTRIBUT PROPRIETE
%token <std::string*>   SI SINON FINSI POUR FINI IDENTIFIANT
%token <int>           ENTIER
%token <std::string*>   CHAINE
%token <std::string*>   HEX_COULEUR RGB_COULEUR
%token <std::string*>   EGAL CROCHET_FERMANT CROCHET_OUVRANT DEUX_POINTS VIRGULE POINT_VIRGULE
%token <std::string*>   PARENTHESE_OUVRANTE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE ACCOLADE_FERMANTE
%token <std::string*>   LARGEUR HAUTEUR COULEURTEXTE COULEURFOND OPACITE


%type <std::variant<int, std::string, std::unique_ptr<Bloc>>> paragraphe element titre sous_titre image titrepage
%type <std::map<std::string, std::map<std::string, std::string>>> attributs
%type <std::map<std::string, std::string>> liste_attributs
%type <std::map<std::string, std::string>> attribut
%type <std::string*> nomattribut
%type <std::string*> valeur define style
%type <std::variant<int, std::string, std::unique_ptr<Bloc>>> valeurvar variable

%%

programme:
    declarations elements
;

elements:
    element elements
    |
;

declarations:
    declaration declarations
    |
;

declaration:
    define
    | style
;

element:
    titre
    | sous_titre
    | paragraphe
    | image
    | titrepage
    | variable
;

titre:
    TITRE attributs CHAINE { 
        $$ = std::make_unique<Titre>($2, $3.texte, 1);
        doc->addBloc(std::move($$));
    }
    | TITRE CHAINE { 
        $$ = std::make_unique<Titre>(std::map<std::string, std::string>(), $2.texte, 1);
        doc->addBloc(std::move($$));
    }
;

sous_titre:
    SOUS_TITRE attributs CHAINE { 
        $$ = std::make_unique<Titre>($2, $3.texte, $3.niveau);
        doc->addBloc(std::move($$));
    }
    | SOUS_TITRE CHAINE { 
        $$ = std::make_unique<Titre>(std::map<std::string, std::string>(), $2.texte, $2.niveau);
        doc->addBloc(std::move($$));
    }
;

paragraphe:
    PARAGRAPHE attributs CHAINE { 
        $$ = std::make_unique<Paragraphe>($2, $3);
        doc->addBloc(std::move($$));
    }
    | PARAGRAPHE CHAINE { 
        $$ = std::make_unique<Paragraphe>(std::map<std::string, std::string>(), $2);
        doc->addBloc(std::move($$));
    }
;

image:
    IMAGE CHAINE { 
        doc->addBloc(std::move(std::make_unique<Image>($2)));
    }
;

attributs:
    CROCHET_OUVRANT liste_attributs CROCHET_FERMANT { 
        $$ = $2;
    }
;

liste_attributs:
    attribut {
        $$ = $1; 
    }
    | attribut VIRGULE liste_attributs {
        $$ = $1;
        $$.insert($3.begin(), $3.end());
    }
    | attribut NEWLINE liste_attributs {
        $$ = $1;
        $$.insert($3.begin(), $3.end());
    }
;

attribut:
    nomattribut DEUX_POINTS valeur { 
         $$ = std::map<std::string, std::string>{{ $1, $3 }};
    }
;

nomattribut:
    LARGEUR { $$ = "width"; }
    | HAUTEUR { $$ = "height"; }
    | COULEURTEXTE { $$ = "color"; }
    | COULEURFOND { $$ = "background-color"; }
    | OPACITE { $$ = "opacity"; }
;

valeur:
    ENTIER { $$ = $1; }
    | HEX_COULEUR { $$ = *$1; }
    | RGB_COULEUR { $$ = *$1; }
    | CHAINE { $$ = *$1; }
    ;

define:
    DEFINE PARENTHESE_OUVRANTE PROPRIETE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE valeur ACCOLADE_FERMANTE 
    { 
        doc->setPropriete(*$3, *$6);
    }
    ;

titrepage:
    TITREPAGE CHAINE { 
        $$ = std::make_unique<TitrePage>($2);
        doc->addBloc(std::move($$));
    }
;

variable:
    IDENTIFIANT EGAL valeurvar { 
        if (std::holds_alternative<std::unique_ptr<Bloc>>($3)) {
            doc->setVariable($1, std::move(std::get<std::unique_ptr<Bloc>>($3)));
        } else if (std::holds_alternative<int>($3)) {
            doc->setVariable($1, std::get<int>($3));
        } else if (std::holds_alternative<std::string>($3)) {
            doc->setVariable($1, std::get<std::string>($3));
        }
    }
;

valeurvar:
    ENTIER { $$ = $1; }
    | HEX_COULEUR { $$ = *$1; }
    | RGB_COULEUR { $$ = *$1; }
    | element { $$ = std::move($1); }
;

style:
    STYLE PARENTHESE_OUVRANTE IDENTIFIANT PARENTHESE_FERMANTE ACCOLADE_OUVRANTE attributs ACCOLADE_FERMANTE 
    { 
        doc->setStyle($3, $6);
    }
;
    
%%

void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur : " << l << ", " << err_msg << std::endl;
}
