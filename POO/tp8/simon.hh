#pragma once

#include "sequence.hh"
#include <QtWidgets>

class simon : public QWidget {
public:
    simon(sequence const & s);
private:
    enum class etat{
        enregistrement,
        restitution
    } _etat;
    QPushButton * _quitter;
    sequence _sequence;
    indicesequence _courant;
    int _joueuractuel;
    std::map<couleur, QPushButton *> _boutonscouleurs;
};
