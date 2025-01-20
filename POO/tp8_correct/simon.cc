#include "simon.hh"
#include <iostream>

simon::simon(sequence const & s)
	:QWidget(),
	 _quitter(new QPushButton("Quitter")),
	 _etat(etat::enregistrement),
	 _courant(0),
	 _sequence(s),
	 _joueuractuel(0) {
	resize(500,350);
	QGridLayout * layoutgeneral = new QGridLayout(this);
	layoutgeneral->addWidget(_quitter, 3,1,1,2);
	_joueurs[0] = new QLineEdit("Joueur 1");
	_joueurs[1] = new QLineEdit("Joueur 2");
	_joueursmarque[0] = new QLabel("");
	_joueursmarque[1] = new QLabel("");
	layoutgeneral->addWidget(_joueursmarque[0],0,1);
	layoutgeneral->addWidget(_joueursmarque[1],1,1);
	layoutgeneral->addWidget(_joueurs[0],0,2);
	layoutgeneral->addWidget(_joueurs[1],1,2);
	layoutgeneral->setRowStretch(2,1);
	layoutgeneral->setColumnStretch(0,1);

	QGridLayout * layoutcouleurs = new QGridLayout();
	layoutgeneral->addLayout(layoutcouleurs, 0,0,4,1);
	_boutonscouleurs[couleur::bleu]=new QPushButton("Bleu");
	_boutonscouleurs[couleur::bleu]->setStyleSheet("QPushButton { background-color: blue; }");
	_boutonscouleurs[couleur::rouge]=new QPushButton("Rouge");
	_boutonscouleurs[couleur::rouge]->setStyleSheet("QPushButton { background-color: red; }");
	_boutonscouleurs[couleur::vert]=new QPushButton("Vert");
	_boutonscouleurs[couleur::vert]->setStyleSheet("QPushButton { background-color: green; }");
	_boutonscouleurs[couleur::jaune]=new QPushButton("Jaune");
	_boutonscouleurs[couleur::jaune]->setStyleSheet("QPushButton { background-color: yellow; }");
	for (auto b : _boutonscouleurs) {
		b.second->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
		connect(b.second, &QPushButton::clicked, this, &simon::oncliccouleur);
	}
	layoutcouleurs->addWidget(_boutonscouleurs[couleur::bleu], 0, 0);
	layoutcouleurs->addWidget(_boutonscouleurs[couleur::rouge], 0, 1);
	layoutcouleurs->addWidget(_boutonscouleurs[couleur::vert], 1, 0);
	layoutcouleurs->addWidget(_boutonscouleurs[couleur::jaune], 1, 1);
	layoutcouleurs->setColumnStretch(0,1);
	layoutcouleurs->setColumnStretch(1,1);

	setWindowTitle("Simon");
	connect(_quitter, &QPushButton::clicked, this, &simon::onclicquitter);
	activerjoueur(0);
}

void simon::onclicquitter() {
	auto quitter = QMessageBox::question(this, "Simon", "Quitter ?", QMessageBox::Yes | QMessageBox::No);
	if (quitter == QMessageBox::Yes)
		close();
}

void simon::oncliccouleur() {
	auto coul(boutonverscouleur(dynamic_cast<QPushButton *>(sender())));
	if (_etat == etat::enregistrement) {
		if (_courant == _sequence.taille()) {
			_sequence.ajouter(coul);
			recommencer();
		}
		else {
			if (_sequence[_courant] != coul)
				perdu();
			else
				_courant++;
		}
	}
}

couleur simon::boutonverscouleur(QPushButton const * b) const {
	return std::find_if(_boutonscouleurs.begin(), _boutonscouleurs.end(), [b](auto const & i){ return i.second == b;})->first;
}

void simon::perdu() {
	std::string msg = "Perdu " + _joueurs[_joueuractuel]->text().toStdString() + ".<br>Longueur atteinte : " + std::to_string(_sequence.taille()) + ".<br>On recommence à zéro !";
	QMessageBox::information(this, "Simon", QString::fromStdString(msg));
	_sequence.vider();
	_courant = 0;
}

void simon::recommencer() {
	_courant = 0;
	QMessageBox::information(this, "Simon", "Bien joué " + _joueurs[_joueuractuel]->text() + ".<br>La main passe !");
	activerjoueur(1-_joueuractuel);
}

void simon::activerjoueur(unsigned char j) {
	_joueuractuel = j;
	_joueursmarque[_joueuractuel]->setText("\u2192");
	_joueursmarque[1-_joueuractuel]->setText(" ");
}

int main(int argc, char *argv[]) {
	QTranslator qtTranslator;
	if (!qtTranslator.load("qt_" + QLocale::system().name(), QLibraryInfo::path(QLibraryInfo::TranslationsPath)))
		return 1;
	QApplication app(argc, argv);
	app.installTranslator(&qtTranslator);
	sequence s;

	if (argc > 1) {
		std::map<std::string, couleur> sc {{"bleu", couleur::bleu}, {"rouge", couleur::rouge}, {"jaune", couleur::jaune}, {"vert", couleur::vert}};
		for (int i=1; i<argc; ++i) {
			if (sc.find(argv[i]) != sc.end())
				s.ajouter(sc[argv[i]]);
			else
				std::cerr << "couleur invalide : " << argv[i] << std::endl;
		}
	}
	simon f(s);
	f.show();
	return app.exec();
}
