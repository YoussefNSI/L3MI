#ifndef DRIVER_H
#define DRIVER_H

#include <string>
#include <functional>
#include "bloc.h"
#include <memory>



class Driver {
public:
    Driver(std::shared_ptr<Document> document);
    ~Driver();
    void executeConditional(bool condition, const std::function<void()>& trueBranch, const std::function<void()>& falseBranch = nullptr);
    std::shared_ptr<Document> getDocument();
    void pushCondition(bool cond) { conditionStack.push(cond); } 
    bool popCondition() {
        bool cond = conditionStack.top();
        conditionStack.pop();
        return cond;
    }
    void resolveConditional() {}
    void setCurrentElseCondition(bool cond) { m_currentElseCondition = cond; }
    bool getCurrentElseCondition() const { return m_currentElseCondition; } // pour recuperer la valeur de la condition dans la règle "else_clause"
private:
    std::stack<bool> conditionStack;
    std::shared_ptr<Document> document;
    bool m_currentElseCondition = false;
};

#endif
