#pragma once

#include <ostream>

using coord = signed short;
class position {
	public:
	position(coord x, coord y);
	coord x() const { return _x; }
	coord y() const { return _y; }
	void setx(coord x) { _x = x; }
	void sety(coord y) { _y = y; }
	bool operator==(position const& p) const;
	bool operator!=(position const& p) const;

	private:
	coord _x;
	coord _y;
};
std::ostream& operator<<(std::ostream& os, position const& p);
