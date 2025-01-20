#include "simon.hh"

simon::simon(const sequence &s)
    :QWidget(),
    _etat(etat::enregistrement),
    _quitter(new QPushButton("Quitter", this)),
    _sequence(s),
    _courant(0),
    _joueuractuel(0) {

    resize(500,350);
    QGridLayout * layoutgeneral(new QGridLayout(this));
    layoutgeneral->addWidget(_quitter, 3, 1 ,1 ,1);
    layoutgeneral->setRowStretch(2, 1);
    layoutgeneral->setColumnStretch(0, 1);

    QGridLayout * layoutcouleurs(new QGridLayout());
    layoutgeneral->addLayout(layoutcouleurs, 0, 0, 4, 1);
    _boutonscouleurs[couleur::bleu] = new QPushButton("bleu");
    _boutonscouleurs[couleur::bleu]->setStyleSheet("QPushButton {background-color: blue;}");
    _boutonscouleurs[couleur::rouge] = new QPushButton("rouge");
    _boutonscouleurs[couleur::rouge]->setStyleSheet("QPushButton {background-color: red;}");
    _boutonscouleurs[couleur::jaune] = new QPushButton("jaune");
    _boutonscouleurs[couleur::jaune]->setStyleSheet("QPushButton {background-color: yellow;}");
    _boutonscouleurs[couleur::vert] = new QPushButton("vert");
    _boutonscouleurs[couleur::vert]->setStyleSheet("QPushButton {background-color: green;}");

    layoutcouleurs->addWidget(_boutonscouleurs[couleur::bleu], 0, 0);
    layoutcouleurs->addWidget(_boutonscouleurs[couleur::rouge], 0, 1);
    layoutcouleurs->addWidget(_boutonscouleurs[couleur::jaune], 1, 0);
    layoutcouleurs->addWidget(_boutonscouleurs[couleur::vert], 1, 1);
    layoutcouleurs->setRowStretch(0, 1);
    layoutcouleurs->setRowStretch(1, 1);

}



