#ifndef TITRE_H
#define TITRE_H

#include "bloc.h"

class Titre : public Bloc {
    std::string texte;
public:
    Titre(const std::string& texte) : texte(texte) {}
    std::string genererHTML() const override {
        return "<h1>" + texte + "</h1>";
    }
};

#endif // TITRE_H