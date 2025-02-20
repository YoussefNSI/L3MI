%skeleton "lalr1.cc"
%require "3.0"

%defines
%define api.parser.class { Parser }
%define api.value.type variant
%define parse.assert
%define parse.error verbose


%locations

%code requires{
    #include "bloc.h"

    class Scanner;
    class Driver;

    struct TitreInfo{
        int niveau;
    };

    extern Document* doc;

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


    #undef  yylex
    #define yylex scanner.yylex
}

%token                 NEWLINE
%token <TitreInfo>     TITRE SOUS_TITRE
%token                 PARAGRAPHE IMAGE
%token                 DEFINE TITREPAGE STYLE
%token <std::string>   PROPRIETE COMMENTAIRE
%token <std::string>   SI SINON FINSI POUR FINI IDENTIFIANT BLOCS
%token <int>           ENTIER
%token <std::string>   CHAINE
%token <std::string>   HEX_COULEUR RGB_COULEUR
%token                 EGAL CROCHET_FERMANT CROCHET_OUVRANT DEUX_POINTS VIRGULE POINT_VIRGULE POINT
%token                 PARENTHESE_OUVRANTE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE ACCOLADE_FERMANTE
%token <std::string>   LARGEUR HAUTEUR COULEURTEXTE COULEURFOND OPACITE
%token <int>           INDICE


%type <Bloc*> bloc_element titre sous_titre paragraphe image titrepage commentaire
%type <std::variant<int, std::string, Bloc*>> variable valeurvar
%type <std::map<std::string, std::string>> attributs
%type <std::map<std::string, std::string>> liste_attributs
%type <std::map<std::string, std::string>> attribut
%type <std::string> nomattribut
%type <std::string> valeur define style
%type <std::pair<std::string, int>> selecteur
%type <int> index_expression

%%

programme:
    programme_element programme
    |
;

programme_element:
    declaration
    | bloc_element
    | variable
    | commentaire
;

declaration:
    define
    | style
    | titrepage
;

bloc_element:
    titre
    | sous_titre
    | paragraphe
    | image
    

titre:
    TITRE attributs CHAINE { 
        $$ = new Titre($2, $3, $1.niveau);
        doc->addBloc("titre", $$);
    }
    | TITRE CHAINE { 
        $$ = new Titre(std::map<std::string, std::string>(), $2, $1.niveau);
        doc->addBloc("titre", $$);
    }
;

sous_titre:
    SOUS_TITRE attributs CHAINE { 
        $$ = new Titre($2, $3, $1.niveau);
        doc->addBloc("titre", $$);
    }
    | SOUS_TITRE CHAINE { 
        $$ = new Titre(std::map<std::string, std::string>(), $2, $1.niveau);
        doc->addBloc("titre", $$);
    }
;

paragraphe:
    PARAGRAPHE attributs CHAINE { 
        $$ = new Paragraphe($2, $3);
        doc->addBloc("paragraphe", $$);
    }
    | PARAGRAPHE CHAINE { 
        $$ = new Paragraphe(std::map<std::string, std::string>(), $2);
        doc->addBloc("paragraphe", $$);
    }
;

image:
    IMAGE CHAINE { 
        doc->addBloc("image", new Image($2));
    }
;

commentaire:
    COMMENTAIRE { 
        doc->addBloc("commentaire", new Commentaire($1));
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
    | HEX_COULEUR { $$ = $1; }
    | RGB_COULEUR { $$ = $1; }
    | CHAINE { $$ = $1; }
    ;

define:
    DEFINE PARENTHESE_OUVRANTE PROPRIETE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE valeur ACCOLADE_FERMANTE 
    { 
        doc->setPropriete($3, $6);
    }
    ;

titrepage:
    TITREPAGE CHAINE { 
        auto bloc = new TitrePage($2);
        doc->addBloc("z", bloc);
    }
;

variable:
    IDENTIFIANT EGAL valeurvar { 
        if (std::holds_alternative<Bloc*>($3)) {
            doc->setVariable($1, std::get<Bloc*>($3));
        } else if (std::holds_alternative<int>($3)) {
            doc->setVariable($1, std::get<int>($3));
        } else if (std::holds_alternative<std::string>($3)) {
            doc->setVariable($1, std::get<std::string>($3));
        }
    }
    | IDENTIFIANT EGAL selecteur { 
        Bloc *b = doc->getNBloc($3.first, $3.second);
        if (b != nullptr) {
            doc->setVariable($1, b);
        }
    }
    IDENTIFIANT POINT nomattribut EGAL valeur {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable($1));
        if (bloc != nullptr) {
            bloc->setPropriete($3.first, $5);
        }
    }
;

selecteur : 
    PARAGRAPHE index_expression { $$ = std::make_pair("p", $2); }
    | TITRE index_expression      { $$ = std::make_pair("h", $2); }
    | SOUS_TITRE index_expression { $$ = std::make_pair("h", $2); }
    | IMAGE index_expression      { $$ = std::make_pair("img", $2); }
;

index_expression:
    INDICE { $$ = $1; }
    | CROCHET_OUVRANT IDENTIFIANT CROCHET_FERMANT {
        auto val = doc->getVariable($2);
        if (!std::holds_alternative<int>(val)) {
            std::cerr << "Erreur: la variable " << $2 << " n'est pas un entier" << std::endl;
            $$ = -2;
        }
        $$ = std::get<int>(val);
    }
;

valeurvar:
    ENTIER { $$ = $1; }
    | HEX_COULEUR { $$ = $1; }
    | RGB_COULEUR { $$ = $1; }
    | bloc_element { 
        $$ = std::variant<int, std::string, Bloc*>($1); 
    }
;

style:
    STYLE PARENTHESE_OUVRANTE BLOCS PARENTHESE_FERMANTE ACCOLADE_OUVRANTE attributs ACCOLADE_FERMANTE 
    { 
        doc->setStyle($3, $6);
    }
;
    
%%

void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur de syntaxe ligne " << l.begin.line 
              << ", colonne " << l.begin.column << ": " << err_msg << std::endl;
}
