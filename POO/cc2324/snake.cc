#include <cstdlib>
#include "snake.hh"

position::position(coord x, coord y) : _x(x), _y(y) {}

coord position::x() const { return _x; }
coord position::y() const { return _y; }

void position::setx(coord x) { _x = x; }

void position::sety(coord y) { _y = y; }

std::string position::tostring() const
{
    return "X" + std::to_string(_x) + "Y" + std::to_string(_y);
}

bool position::egal(position const & p) const
{
    return _x == p.x() && _y == p.y();
}

bool position::adjacent(position const & p) const
{
    auto dx(static_cast<signed int> (p.x()) - static_cast<signed int> (_x));
    auto dy(static_cast<signed int> (p.y()) - static_cast<signed int> (_y));
    return ((dy == 0) && (std::abs(dx) == 1)) || ((std::abs(dy) == 1) && (dx == 0));
}

