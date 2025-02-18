#ifndef PARAGRAPHE_H
#define PARAGRAPHE_H

#include "bloc.h"

class Paragraphe : public Bloc {
    std::string texte;
public:
    Paragraphe(const std::string& texte) : texte(texte) {}
    std::string genererHTML() const override {
        return "<p>" + texte + "</p>";
    }
};

#endif // PARAGRAPHE_H