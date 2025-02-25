#include <functional>
#include <stack>
#include "driver.hh"
void Driver::pushCondition(bool cond) {
    conditionStack.push(cond);
}

Driver::~Driver() {
    while (!conditionStack.empty()) {
        conditionStack.pop();
    }
}

bool Driver::popCondition() {
    bool cond = conditionStack.top();
    conditionStack.pop();
    return cond;
}