#ifndef SCANNER_H
#define SCANNER_H

#if ! defined(yyFlexLexerOnce)
#include <FlexLexer.h>
#endif


class Scanner : public yyFlexLexer {
public:
    unsigned int nbMots=0;
    unsigned int nbNombres=0;
    unsigned int nbPonctuations=0;

    void affichage(){
        std::cout << "Il y a : " << std::endl;
        std::cout << "\t" << nbMots << " mots" << std::endl;
        std::cout << "\t" << nbNombres << " nombres" << std::endl;
        std::cout << "\t" << nbPonctuations << " signes de ponctuation" << std::endl;
    }

    Scanner(std::istream & in, std::ostream & out) : yyFlexLexer(in, out) {
    }

    ~Scanner() {
    }

    virtual
    int yylex();


    // ajouter des variables/méthodes si besoin (exercice avec compteur de mots)

};


#endif
