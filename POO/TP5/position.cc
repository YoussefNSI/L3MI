#include "position.hh"

position::position(coord x, coord y)
	: _x(x)
	, _y(y) { }

bool position::operator==(const position& p) const {
	return (_x == p._x) && (_y == p._y);
}

bool position::operator!=(const position& p) const {
	return !operator==(p);
}

std::ostream& operator<<(std::ostream& os, const position& p) {
	os << "(" << p.x() << "," << p.y() << ")";
	return os;
}
