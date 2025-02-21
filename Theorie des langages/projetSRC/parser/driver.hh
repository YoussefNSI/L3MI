#ifndef DRIVER_H
#define DRIVER_H

#include <string>


class Driver {
private:
    std::string instructions;       

public:
    Driver();
    ~Driver();
    std::string executeInstruction(std::string instruction);
    std::string getInstructions();
    
   
};

#endif
