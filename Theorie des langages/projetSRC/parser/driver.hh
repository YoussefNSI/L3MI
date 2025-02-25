#ifndef DRIVER_H
#define DRIVER_H

#include <string>
#include <functional>



class Driver {
public:
    Driver() = default;
    ~Driver();
    void pushCondition(bool cond); 
    bool popCondition();
    void resolveConditional() {}
    void setCurrentElseCondition(bool cond) { m_currentElseCondition = cond; }
    bool getCurrentElseCondition() const { return m_currentElseCondition; } // pour recuperer la valeur de la condition dans la règle "else_clause"
private:
    std::stack<bool> conditionStack;
    bool m_currentElseCondition = false;
};

#endif
