#pragma once
#include <vector>
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
    sequence() =default;
    sequence(sequence const & s);
    ~sequence() =default;
    void ajouter(couleur c);
    indicesequence taille() const;
    void afficher(couleur c) const;
    void afficher(couleur c, std::ostream & os) const;
    void vider();
    void afficher(std::ostream & os) const;
    sequence & operator=(sequence const & s){
        _contenu = s._contenu;
        return *this;
    }
    bool operator==(sequence const & s) const {
        return _contenu == s._contenu;
    }
    couleur operator[](indicesequence i) const {
        return _contenu.at(i);
    }
private:
    std::vector<couleur> _contenu;
};

std::ostream & operator<<(std::ostream & os, couleur c);
std::ostream & operator<<(std::ostream & os, sequence const & s){
    s.afficher(os);
    return os;
}
