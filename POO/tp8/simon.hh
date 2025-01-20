#pragma once

#include "sequence.hh"
#include <QtWidgets>

class simon : public QWidget {
public:
    simon(sequence const & s);
public slots:
    void onclicquitter();
    void oncliccouleur();
private:
    enum class etat{
        enregistrement,
        restitution
    } _etat;
    QPushButton * _quitter;
    sequence _sequence;
    sequence::indice _courant;
    int _joueuractuel;
    std::map<couleur, QPushButton *> _boutonscouleurs;

    couleur boutonverscouleur(QPushButton * b) const;
};
