#include "banque.hh"


virement::virement(std::shared_ptr<compte> source, std::shared_ptr<compte> destination, float montant)
	: _source(source)
	, _destination(destination)
	, _montant(montant) {
}

void banque::appliquerinterets() {
	for (auto& i : _comptes)
		i->appliquerinterets();
}

std::shared_ptr<proprietaire> banque::chercheproprietaire(const std::string& n) {
	for (auto& i : _proprietaires)
		if (i->nom().find(n) != std::string::npos)
			return i;
	return nullptr;
}

std::vector<unsigned int> banque::comptes_numero(std::shared_ptr<proprietaire> p) const {
	std::vector<unsigned int> result;
	for (auto& i : _comptes)
		if ((i->prop() == p) || (i->prop()->lie_a(p)))
			result.push_back(i->numero());
	return result;
}

std::vector<std::shared_ptr<compte>> banque::comptes_de(std::shared_ptr<proprietaire> p) const {
	std::vector<std::shared_ptr<compte>> result;
	for (auto& i : _comptes)
		if (i->prop() == p)
			result.push_back(i);
	return result;
}

std::vector<std::shared_ptr<compte>> banque::comptes_decouvert() const {
	std::vector<std::shared_ptr<compte>> result;
	for (auto& i : _comptes)
		if (i->montant() < 0)
			result.push_back(i);
	return result;
}

float banque::sommetotale(std::shared_ptr<proprietaire> p) const {
	float result(0);
	for (auto& i : _comptes)
		if (p == i->prop())
			result += i->montant();
	return result;
}

void banque::appliquervirements() {
	for (auto const& i : _virementsauto)
		appliquervirement(i);
}

void banque::appliquervirement(const virement& v) {
	v.source()->virer(*(v.destination()), v.montant());
}
