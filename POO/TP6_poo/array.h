#ifndef ARRAY_H
#define ARRAY_H

#endif // ARRAY_H

#include <vector>
#include <exception>
#include <string>

class arrayint
{
public:
    arrayint(int imin, int imax)
        : _imin(imin), _imax(imax)
    {
        _tab.resize(_imax - _imin + 1);
    }
    arrayint(arrayint const &a) = default;
    int get_imin() const { return _imin; }
    int get_imax() const { return _imax; }
    int at(int i) const;
    int operator[](int i) const;
    void set(int i, int val);
    bool operator==(arrayint const &a) const;
    arrayint &operator=(arrayint const &a);

private:
    std::vector<int> _tab;
    int _imin;
    int _imax;
};

class exceptionarrayint : public std::exception
{
public:
    exceptionarrayint(std::string msg, int i)
        : _msg(msg), _i(i) {}
    const char* what() const noexcept override { return _msg.c_str(); }
    int get_i() const { return _i; }

private:
    const std::string _msg;
    int _i;
};
