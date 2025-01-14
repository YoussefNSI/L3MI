#pragma once

#include "proprietaire.hh"
#include "compte.hh"
#include <memory>
#include <vector>

class virement {
	public:
	virement(std::shared_ptr<compte> source, std::shared_ptr<compte> destination, float montant);
	std::shared_ptr<compte> source() const { return _source; }
	std::shared_ptr<compte> destination() const { return _destination; }
	float montant() const { return _montant; }

	private:
	std::shared_ptr<compte> _source;
	std::shared_ptr<compte> _destination;
	float _montant;
};

class banque {
	public:
	banque() = default;
	banque(banque const&) = delete;
	banque& operator=(banque const&) = delete;

	void appliquerinterets();
	std::shared_ptr<proprietaire> chercheproprietaire(std::string const& n);
	std::vector<unsigned int> comptes_numero(std::shared_ptr<proprietaire> p) const;
	std::vector<std::shared_ptr<compte>> comptes_de(std::shared_ptr<proprietaire> p) const;
	std::vector<std::shared_ptr<compte>> comptes_decouvert() const;
	float sommetotale(std::shared_ptr<proprietaire> p) const;

	void ajouter_virementauto(virement const& v) { _virementsauto.push_back(v); }
	void appliquervirements();

	private:
	void appliquervirement(virement const& v);

	private:
	std::vector<std::shared_ptr<proprietaire>> _proprietaires;
	std::vector<std::shared_ptr<compte>> _comptes;
	std::vector<virement> _virementsauto;
};
