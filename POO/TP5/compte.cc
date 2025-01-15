#include "compte.hh"

unsigned int compte::_compteur(0);

bool compte::verser(float m) {
	if (montantautorise(_montant+m)) {
		_montant += m;
		return true;
	}
	else
		return false;
}

bool compte::retirer(float m) {
	if (montantautorise(_montant-m)) {
		_montant -= m;
		return true;
	}
	else
		return false;
}

bool compte::virer(compte & dest, float m) {
	float commission((_prop == dest._prop) ? 0 : 1);
	if (montantautorise(_montant-m-commission) && dest.montantautorise(dest._montant+m)) {
		_montant -= m+commission;
		dest._montant += m;
		return true;
	}
	else
		return false;
}

void compte::appliquerinterets() {
	_montant += _montant*taux();
}

float compte::taux() const {
	return 0;
}

float ldd::_taux(0.0075);
float lep::_taux(0.0125);
