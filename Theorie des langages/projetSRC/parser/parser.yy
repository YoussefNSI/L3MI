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
%token                 DEFINE TITREPAGE STYLE SELECTSTYLE
%token <std::string>   PROPRIETE COMMENTAIRE
%token <std::string>   SI SINON FINSI POUR FINI IDENTIFIANT BLOCS
%token <int>           ENTIER
%token <std::string>   CHAINE
%token <std::string>   HEX_COULEUR RGB_COULEUR
%token                 EGAL CROCHET_FERMANT CROCHET_OUVRANT DEUX_POINTS VIRGULE POINT_VIRGULE POINT PLUS MOINS MULT DIV
%token                 PARENTHESE_OUVRANTE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE ACCOLADE_FERMANTE
%token                 LARGEUR HAUTEUR COULEURTEXTE COULEURFOND OPACITE


%type <Bloc*> bloc_element titre sous_titre paragraphe image titrepage commentaire
%type <std::variant<int, std::string, Bloc*, std::map<std::string, std::string>>> variable valeurvar
%type <std::map<std::string, std::string>> attributs
%type <std::map<std::string, std::string>> liste_attributs
%type <std::map<std::string, std::string>> attribut
%type <std::string> nomattribut
%type <std::string> valeur define style
%type <std::pair<std::string, int>> selecteur
%type <int> index_expression expr terme facteur

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
        doc->addBloc($$);
    }
    | TITRE CHAINE { 
        $$ = new Titre(std::map<std::string, std::string>(), $2, $1.niveau);
        doc->addBloc($$);
    }
;

sous_titre:
    SOUS_TITRE attributs CHAINE { 
        $$ = new Titre($2, $3, $1.niveau);
        doc->addBloc($$);
    }
    | SOUS_TITRE CHAINE { 
        $$ = new Titre(std::map<std::string, std::string>(), $2, $1.niveau);
        doc->addBloc($$);
    }
;

paragraphe:
    PARAGRAPHE attributs CHAINE { 
        $$ = new Paragraphe($2, $3);
        doc->addBloc($$);
    }
    | PARAGRAPHE CHAINE { 
        $$ = new Paragraphe(std::map<std::string, std::string>(), $2);
        doc->addBloc($$);
    }
;

image:
    IMAGE CHAINE { 
        doc->addBloc(new Image($2));
    }
;

commentaire:
    COMMENTAIRE { 
        doc->addBloc(new Commentaire($1));
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
    | attribut liste_attributs {
        $$ = $1;
        $$.insert($2.begin(), $2.end());
    }
;

attribut:
    nomattribut DEUX_POINTS valeur { 
         $$ = std::map<std::string, std::string>{{ $1, $3 }};
    }
    | nomattribut DEUX_POINTS IDENTIFIANT {
        std::string val = std::get<std::string>(doc->getVariable($3));
        if (val != "") {
            $$ = std::map<std::string, std::string>{{ $1, val }};
        }
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
    ENTIER { $$ = std::to_string($1); } // ne pas oublier de gerer ce cas dans bloc.cc/bloc.h
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
        doc->addBloc(bloc);
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
        } else if (std::holds_alternative<std::map<std::string, std::string>>($3)) {
            doc->setVariable($1, std::get<std::map<std::string, std::string>>($3));
        }

    }
    | IDENTIFIANT EGAL selecteur { 
        Bloc *b = doc->getNBloc($3.first, $3.second);
        if (b != nullptr) {
            doc->setVariable($1, b);
        }
    }
    | IDENTIFIANT POINT nomattribut EGAL valeur {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable($1));
        if (bloc != nullptr) {
            bloc->setPropriete($3, $5);
        }
    }
    | IDENTIFIANT POINT nomattribut EGAL IDENTIFIANT {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable($1));
        if (bloc != nullptr) {
            std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> prop = doc->getVariable($5);
            if (std::holds_alternative<std::string>(prop)) {
                bloc->setPropriete($3, std::get<std::string>(prop));
            }
        }
    }
    | IDENTIFIANT POINT SELECTSTYLE EGAL IDENTIFIANT {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable($1));
        if (bloc != nullptr) {
            std::variant<int, std::string, Bloc*, std::map<std::string, std::string>> prop = doc->getVariable($5);
            if (std::holds_alternative<std::map<std::string, std::string>>(prop)) {
                for (auto const& [key, val] : std::get<std::map<std::string, std::string>>(prop)) {
                    bloc->setPropriete(key, val);
                }
            }
        }
    }
    | IDENTIFIANT POINT SELECTSTYLE EGAL attributs {
        Bloc* bloc = std::get<Bloc*>(doc->getVariable($1));
        if (bloc != nullptr) {
            for (auto const& [key, val] : $5) {
                bloc->setPropriete(key, val);
            }
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
    CROCHET_OUVRANT expr CROCHET_FERMANT { $$ = $2; }
;

expr:
    expr PLUS terme { $$ = $1 + $3; }
    | expr MOINS terme { $$ = $1 - $3; }
    | terme { $$ = $1; }
;

terme:
    terme MULT facteur { $$ = $1 * $3; }
    | terme DIV facteur { $$ = $1 / $3; }
    | facteur { $$ = $1; }
;

facteur:
    ENTIER { $$ = $1; }
    | IDENTIFIANT {
        auto val = doc->getVariable($1);
        if(std::holds_alternative<int>(val)) {
            $$ = std::get<int>(val);
        } else {
            std::cerr << "Erreur: la variable " << $1 << " n'est pas un entier" << std::endl;
            $$ = -1;
        }
    }

valeurvar:
    ENTIER { $$ = $1; }
    | HEX_COULEUR { $$ = $1; }
    | RGB_COULEUR { $$ = $1; }
    | bloc_element { $$ = std::variant<int, std::string, Bloc*, std::map<std::string, std::string>>($1); }
    | attributs { $$ = $1; }
;

style:
    STYLE PARENTHESE_OUVRANTE BLOCS PARENTHESE_FERMANTE CROCHET_OUVRANT liste_attributs CROCHET_FERMANT 
    { 
        doc->setStyle($3, $6);
    }
;
    
%%

void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur de syntaxe ligne " << l.begin.line 
              << ", colonne " << l.begin.column << ": " << err_msg << std::endl;
}
