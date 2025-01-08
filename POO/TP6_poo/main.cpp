#include <iostream>
#include "array.h"


int main()
{
    arrayint a(0, 5);
    a.set(0, 1);
    a.set(1, 2);
    a.set(2, 3);
    std::cout << "at 0 : " << a.at(0) << std::endl;
    std::cout << "at 1 : " << a.at(1) << std::endl;
    std::cout << "[2] : " << a[2] << std::endl;
    std::cout << "imin : " << a.get_imin() << std::endl;
    std::cout << "imax : " << a.get_imax() << std::endl;

    return 0;
}
