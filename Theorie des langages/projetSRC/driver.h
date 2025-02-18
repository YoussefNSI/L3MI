#ifndef DRIVER_H
#define DRIVER_H

#include "contexte.h"

class Driver {
    Contexte contexte;
public:
    Contexte& getContexte() { return contexte; }
};

#endif // DRIVER_H