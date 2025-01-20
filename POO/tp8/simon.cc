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
    _boutonscouleurs[couleur::bleu] = new QPushButton("Bleu");
    _boutonscouleurs[couleur::bleu]->setStyleSheet("QPushButton {background-color: blue;}");
    _boutonscouleurs[couleur::rouge] = new QPushButton("Rouge");
    _boutonscouleurs[couleur::rouge]->setStyleSheet("QPushButton {background-color: red;}");
    _boutonscouleurs[couleur::jaune] = new QPushButton("Jaune");
    _boutonscouleurs[couleur::jaune]->setStyleSheet("QPushButton {background-color: yellow;}");
    _boutonscouleurs[couleur::vert] = new QPushButton("Vert");
    _boutonscouleurs[couleur::vert]->setStyleSheet("QPushButton {background-color: green;}");
    for(auto b : _boutonscouleurs) {
        b.second->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        connect(b.second, &QPushButton::clicked, this, &simon::oncliccouleur);
    }

    layoutcouleurs->addWidget(_boutonscouleurs[couleur::bleu], 0, 0);
    layoutcouleurs->addWidget(_boutonscouleurs[couleur::rouge], 0, 1);
    layoutcouleurs->addWidget(_boutonscouleurs[couleur::jaune], 1, 0);
    layoutcouleurs->addWidget(_boutonscouleurs[couleur::vert], 1, 1);
    layoutcouleurs->setRowStretch(0, 1);
    layoutcouleurs->setRowStretch(1, 1);

    setWindowTitle("Simon");
    connect(_quitter, &QPushButton::clicked, this, &simon::onclicquitter);

}

couleur simon::boutonverscouleur(QPushButton * b) const {
    return std::find_if(_boutonscouleurs.begin(), _boutonscouleurs.end(), [b](auto p){ return p.second == b; })->first;
}

void simon::onclicquitter() {
    auto quitter = QMessageBox::question(this, "Quitter", "Voulez-vous vraiment quitter ?", QMessageBox::Yes | QMessageBox::No);
    if (quitter == QMessageBox::Yes){
        close();
    }
}

void simon::oncliccouleur(){
    auto coul(boutonverscouleur(dynamic_cast<QPushButton *>(sender())));
    if(_etat == etat::enregistrement){
        if(_courant == _sequence.taille()){
            _sequence.ajouter(coul);
        }
    }
}



