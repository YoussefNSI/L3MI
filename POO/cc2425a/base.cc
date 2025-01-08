#include "base.hh"
#include <iostream>
void base_messages::afficher_apres(const horaire& h) const {
	for (auto const& i : _res)
		if (horaire::est_avant(h, i.horaire_choisi())) {
			i.sortie_flux(std::cout);
			std::cout << "\n";
		}
}

bool base_messages::repondre(idmessage p, personne const& d, const horaire& h) {
	auto propit(rechercher_prop(p));
	if (propit == _prop.end())
		return false;
	if (!propit->contient_destinataire(d))
		return false;
	if (!propit->contient(h))
		return false;
	std::list<personne> dest;
	dest.push_back(propit->expediteur());
	ajouter(reponse(d, dest, p, h));
	return true;
}

bool base_messages::verifier_reponses() const {
	for (auto const& r : _rep) {
		auto propit(rechercher_prop(r.suite_de()));
		if (propit == _prop.end())
			return false;
		if (!propit->contient(r.preference()))
			return false;
	}
	return true;
}

void base_messages::purger() {
	for (auto it(_rep.begin()); it != _rep.end();) {
		if (contient_resultat(it->suite_de()))
			it = _rep.erase(it);
		else
			++it;
	}
}

void base_messages::supprimer(idmessage p) {
	for (auto i(_prop.begin()); i != _prop.end(); ++i) {
		if (i->id() == p) {
			_prop.erase(i);
			for (auto j(_rep.begin()); j != _rep.end();) {
				if (j->suite_a(p))
					j = _rep.erase(j);
				else
					++j;
			}
			for (auto j(_res.begin()); j != _res.end();) {
				if (j->suite_a(p))
					j = _res.erase(j);
				else
					++j;
			}
			return;
		}
	}
	for (auto i(_rep.begin()); i != _rep.end(); ++i)
		if (i->id() == p) {
			_rep.erase(i);
			return;
		}
	for (auto i(_res.begin()); i != _res.end(); ++i)
		if (i->id() == p) {
			_res.erase(i);
			return;
		}
}

std::list<proposition>::const_iterator base_messages::rechercher_prop(idmessage idp) const {
	for (auto i(_prop.begin()); i != _prop.end(); ++i)
		if (i->id() == idp)
			return i;
	return _prop.end();
}

bool base_messages::contient_resultat(idmessage i) const {
	for (auto const & r : _res)
		if (r.id() == i)
			return true;
	return false;
}
