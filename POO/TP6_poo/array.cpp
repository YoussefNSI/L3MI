#include "array.h"

int arrayint::at(int i) const
{
    if (i < _imin || i > _imax)
        throw exceptionarrayint("index out of range", i);
    return _tab[i - _imin];
}

int arrayint::operator[](int i) const
{
    if (i < _imin || i > _imax)
        throw exceptionarrayint("index out of range", i);
    return _tab[i - _imin];
}

void arrayint::set(int i, int val)
{
    if (i < _imin || i > _imax)
        throw exceptionarrayint("index out of range", i);
    _tab[i - _imin] = val;
}

bool arrayint::operator==(arrayint const &a) const
{
    if (_imin != a._imin || _imax != a._imax)
        return false;
    return _tab == a._tab;
}

arrayint &arrayint::operator=(arrayint const &a)
{
    if (this != &a)
    {
        _imin = a._imin;
        _imax = a._imax;
        _tab = a._tab;
    }
    return *this;
}

std::ostream &operator<<(std::ostream &os, arrayint const &a)
{
    int imin = a.get_imin();
    try
    {
        os << "[";
        os << a.at(imin);
        for (int i = imin + 1; i <= a.get_imax(); ++i)
            os << ", " << a.at(i);
        os << "]";
    }
    catch (exceptionarrayint const &e)
    {
        ;
    }
    return os;
}
