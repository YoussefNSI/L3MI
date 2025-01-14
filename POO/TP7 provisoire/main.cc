#include <iostream>
#include "ensemble.hh"


int main()
{
    ensemble<int> e1;
    ensemble<int> e2;
    e1.insert(1);
    e1.insert(2);
    e1.insert(3);
    e2.insert(4);
    e2.insert(5);
    e1.unionwith(e2);
    std::cout << "find 1 " << e1.find(1) << std::endl;
    std::cout << "find 2 " << e1.find(2) << std::endl;
    std::cout << "find 3 " << e1.find(3) << std::endl;
    std::cout << "find 4 " << e1.find(4) << std::endl;
    for(auto i : e1.get_tab()) {
        std::cout << i << std::endl;};
    return 0;
}
