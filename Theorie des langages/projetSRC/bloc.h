#ifndef BLOC_H
#define BLOC_H

#include <string>

class Bloc {
public:
    virtual ~Bloc() = default;
    virtual std::string genererHTML() const = 0;
};

#endif // BLOC_H