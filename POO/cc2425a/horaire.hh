#pragma once
#include <string>

class horaire {
	public:
	horaire(unsigned short a, unsigned short m, unsigned short j, unsigned short h)
		: _annee(a)
		, _mois(m)
		, _jour(j)
		, _heure(h) { }
	unsigned short annee() const { return _annee; }
	unsigned short mois() const { return _mois; }
	unsigned short jour() const { return _jour; }
	unsigned short heure() const { return _heure; }
	std::string to_string() const;
	bool concerne_amj(unsigned short a, unsigned short m, unsigned short j) const;
	static bool est_avant(horaire const& h1, horaire const& h2);
	static bool est_egal(horaire const& h1, horaire const& h2);
	private:
	unsigned short _annee;
	unsigned short _mois;
	unsigned short _jour;
	unsigned short _heure;
};
