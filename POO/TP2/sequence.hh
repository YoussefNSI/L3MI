#ifndef SEQUENCE_HH
#define SEQUENCE_HH

#endif // SEQUENCE_HH

#pragma once
#include <cstddef>
#include <iostream>

enum class couleur {
    rouge,
    bleu,
    jaune,
    vert
};

using indicesequence = std::size_t;

class sequence {
public:
    sequence();
    sequence(sequence const & s);
    ~sequence();
    void ajouter(couleur c);
    indicesequence taille() const;
    void afficher(couleur c) const;
    void afficher(couleur c, std::ostream & os) const;
    couleur acces(indicesequence i) const;
    void vider();
    void afficher(std::ostream & os) const;
    bool comparer(sequence const & s) const;
    void copier(sequence const & s);
private:
    couleur * _contenu;
    indicesequence _taille;
};
