#include "echiquier.hh"
#include <algorithm>

echiquier::echiquier(echiquier const & e) {
	for (auto const & p : e._pieces)
		_pieces.push_back(p->clone());
}

echiquier & echiquier::operator=(echiquier const & e) {
	if (this != &e) {
		_pieces.clear();
		for (auto const & p : e._pieces)
			_pieces.push_back(p->clone());
	}
	return *this;
}

unsigned int echiquier::valeurdujoueur(couleur c) const {
	unsigned int result(0);
	for (auto const & p : _pieces) {
		if (p->coul() == c)
			result += p->valeur();
	}
	return result;
}

bool echiquier::deplacer(position const & src, position const & dst) {
	auto fsrc(std::find_if(_pieces.begin(), _pieces.end(), [&src](auto const & p){return p->pos() == src;}));
	if (fsrc == _pieces.end())
		return false;
	auto fdst(std::find_if(_pieces.begin(), _pieces.end(), [&dst](auto const & p){return p->pos() == dst;}));
	if (fdst == _pieces.end())
		return (**fsrc).deplacer(dst);
	else {
		if ((**fsrc).coul() != (**fdst).coul())
			if ((**fsrc).deplacer(dst)) {
				_pieces.erase(fdst);
				return true;
			}
		return false;
	}
}

void echiquier::affichage(std::ostream & os) const {
	for (coord y=0; y<8; ++y) {
		for (coord x=0; x<8; ++x) {
			auto f(std::find_if(_pieces.begin(), _pieces.end(), [x,y](auto const & p){return (p->pos().x() == x) && (p->pos().y() == y);}));
			if (f == _pieces.end())
				os << "  ";
			else
				os << std::string() + (*f)->symbole() + (((*f)->coul() == couleur::noir) ? "N" : "B");
		}
		os << "\n";
	}
}

void echiquier::sauvegarde(std::ofstream & os) const {
	for (auto const & p : _pieces)
		os << p->tostring() << "\n";
}

void echiquier::chargement(std::ifstream & os) {
	while (!os.eof()) {
		std::string line;
		os >> line;
		if (line.empty())
			break;
		position p(line[2]-'0', line[3]-'0');
		couleur c(line[1] == 'N' ? couleur::noir : couleur::blanc);
		std::unique_ptr<piece> np;
		switch (line[0]) {
			case 'P':
				np = std::make_unique<pion>(p, c); break;
			case 'C':
				np = std::make_unique<cavalier>(p, c); break;
			case 'D':
				np = std::make_unique<dame>(p, c); break;
			case 'R':
				np = std::make_unique<roi>(p, c); break;
		}
		_pieces.push_back(std::move(np));
	}
}

void echiquier::depart() {
	_pieces.clear();
	ajout(std::make_unique<cavalier>(position(1,0), couleur::noir));
	ajout(std::make_unique<dame>(position(3,0), couleur::noir));
	ajout(std::make_unique<roi>(position(4,0), couleur::noir));
	ajout(std::make_unique<cavalier>(position(6,0), couleur::noir));
	for (coord x=0; x<8; ++x) {
		ajout(std::make_unique<pion>(position(x,1), couleur::noir));
		ajout(std::make_unique<pion>(position(x,6), couleur::blanc));
	}
	ajout(std::make_unique<cavalier>(position(1,7), couleur::blanc));
	ajout(std::make_unique<dame>(position(3,7), couleur::blanc));
	ajout(std::make_unique<roi>(position(4,7), couleur::blanc));
	ajout(std::make_unique<cavalier>(position(6,7), couleur::blanc));
}

bool echiquier::aperdu(couleur c) const {
	return std::none_of(_pieces.begin(), _pieces.end(), [c](auto const & p){return ((c==p->coul()) && (p->symbole()=='R'));});
}

bool echiquier::contientpiececouleur(position const & p, couleur c) {
	auto const & f(std::find_if(_pieces.begin(), _pieces.end(), [&p](auto const & pc){ return p == pc->pos();}));
	return (f != _pieces.end()) && ((*f)->coul() == c);
}
