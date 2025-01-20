#include "sequence.hh"
#include <algorithm>

std::ostream & operator<<(std::ostream & os, couleur c) {
	switch (c) {
		case couleur::bleu: os << "bleu"; break;
		case couleur::rouge: os << "rouge"; break;
		case couleur::jaune: os << "jaune"; break;
		case couleur::vert: os << "vert"; break;
	}
	return os;
}

std::ostream & operator<<(std::ostream & os, sequence const & s) {
	std::for_each(s._couleurs.begin(), s._couleurs.end(), [&os](auto c){ os << c << " "; });
	return os;
}
