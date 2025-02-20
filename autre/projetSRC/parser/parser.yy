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
%token <std::string>   SI SINON FINSI POUR FINI IDENTIFIANT
%token <int>           ENTIER
%token <std::string>   CHAINE
%token <std::string>   HEX_COULEUR RGB_COULEUR
%token <std::string>   EGAL CROCHET_FERMANT CROCHET_OUVRANT DEUX_POINTS VIRGULE POINT_VIRGULE
%token                 PARENTHESE_OUVRANTE PARENTHESE_FERMANTE ACCOLADE_OUVRANTE ACCOLADE_FERMANTE
%token <std::string>   LARGEUR HAUTEUR COULEURTEXTE COULEURFOND OPACITE


%type <Bloc*> bloc_element titre sous_titre paragraphe image titrepage commentaire
%type <std::variant<int, std::string, Bloc*>> variable valeurvar
%type <std::map<std::string, std::string>> attributs
%type <std::map<std::string, std::string>> liste_attributs
%type <std::map<std::string, std::string>> attribut
%type <std::string> nomattribut
%type <std::string> valeur define style

%%

programme:
    programme_element programme
    |
;

programme_element:
    declaration
    | bloc_element
    | variable

declaration:
    define
    | style
;

bloc_element:
    titre
    | sous_titre
    | paragraphe
    | image
    | titrepage
    | commentaire
;

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
        }
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
    STYLE PARENTHESE_OUVRANTE IDENTIFIANT PARENTHESE_FERMANTE ACCOLADE_OUVRANTE attributs ACCOLADE_FERMANTE 
    { 
        doc->setStyle($3, $6);
    }
;
    
%%

void yy::Parser::error( const location_type &l, const std::string & err_msg) {
    std::cerr << "Erreur de syntaxe ligne " << l.begin.line 
              << ", colonne " << l.begin.column << ": " << err_msg << std::endl;
}
