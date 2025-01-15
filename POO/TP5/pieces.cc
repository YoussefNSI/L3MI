#include "pieces.hh"
#include <algorithm>

std::ostream & operator<<(std::ostream & os, position const & p) {
	os << "(" << p.x() << "," << p.y() << ")";
	return os;
}

piece::~piece() {
}

bool piece::accepterposition(position const & p) const {
	auto dp(deplacementspossibles());
	return std::find(dp.begin(), dp.end(), p) != dp.end();
}

bool piece::deplacer(position const & p) {
	if (accepterposition(p)) {
		_position = p;
		return true;
	}
	else
		return false;
}

std::vector<position> pion::deplacementspossibles() const {
	std::vector<position> res;
	auto p(pos());
	p.sety(p.y() + ((coul() == couleur::noir) ? 1 : -1));
	if (p.estvalide())
		res.push_back(p);
	return res;
}

std::vector<position> roi::deplacementspossibles() const {
	std::vector<position> res;
	auto const & actuel = pos();
	for (coord x(-1); x<=1; ++x)
		for (coord y(-1); y<=1; ++y)
			if ((x != 0) || (y != 0)) {
				position p(actuel.x()+x, actuel.y()+y);
				if (p.estvalide())
					res.push_back(p);
			}
	return res;
}

std::vector<position> dame::deplacementspossibles() const {
	std::vector<position> res;
	auto const & actuel = pos();
	for (coord x(-1); x<=1; ++x)
		for (coord y(-1); y<=1; ++y)
			if ((x != 0) || (y != 0))
				for (coord i=1; i<=7; ++i) {
					position p(actuel.x() + x*i, actuel.y() + y*i);
					if (p.estvalide())
						res.push_back(p);
					else
						break;
				}
	return res;
}

std::vector<position> cavalier::deplacementspossibles() const {
	std::vector<position> res;
	auto const & actuel = pos();
	for (coord a : {-2, 2})
		for (coord b : {-1, 1}) {
			position p1(actuel.x() + a, actuel.y() + b);
			if (p1.estvalide())
				res.push_back(p1);
			position p2(actuel.x() + b, actuel.y() + a);
			if (p2.estvalide())
				res.push_back(p2);
		}
	return res;
}
