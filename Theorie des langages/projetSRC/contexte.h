#ifndef CONTEXTE_H
#define CONTEXTE_H

#include <map>
#include <string>

class Contexte {
    std::map<std::string, std::string> variables;
public:
    void setVariable(const std::string& nom, const std::string& valeur);
    std::string getVariable(const std::string& nom) const;
};

#endif // CONTEXTE_H