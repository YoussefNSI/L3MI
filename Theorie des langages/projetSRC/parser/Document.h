#pragma once

#include "bloc.h"
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <variant>
#include <set>
#include <stack>

using VariableType = std::variant<int, std::string, std::shared_ptr<Bloc>, std::map<std::string, std::string>>;

class Document
{
public:
    Document();
    ~Document() = default;
    void addBloc(std::shared_ptr<Bloc> bloc);
    void setPropriete(const std::string &nom, const std::string &valeur)
    {
        proprietes[nom] = valeur;
    }
    void setVariable(const std::string &nom, VariableType valeur)
    {
        variables[nom] = valeur;
    }
    void setStyle(const std::string &nom, std::map<std::string, std::string> valeur)
    {
        mapStyles[nom] = valeur;
    }
    void afficherBlocs() const; // Méthode pour le débogage seulement
    std::string getPropriete(const std::string &nom) const;
    const VariableType &getVariable(const std::string &nom) const;
    std::map<std::string, std::string> getStyle(const std::string &nom) const;
    std::string toHTML() const;
    void HTMLtoFile(const std::string &filename) const;

    std::shared_ptr<Bloc> getNBloc(const std::string &type, int index) const;
    void beginTransaction();
    void commitTransaction();
    void rollbackTransaction();

private:
    std::map<std::string, std::shared_ptr<Bloc>> blocs;                                // blocs
    std::pair<std::string, std::shared_ptr<Bloc>> metablocs;                           // titrepage
    std::map<std::string, std::string> proprietes;                       // @DEFINE
    std::map<std::string, VariableType> variables;                       // variables
    std::map<std::string, std::map<std::string, std::string>> mapStyles; // @STYLE
    std::map<std::string, int> blocCounts;                             // compteur de bloc pour chaque type

    struct TransactionState {
        std::map<std::string, VariableType> variables;
        std::map<std::string, std::string> proprietes;
        std::map<std::string, std::map<std::string, std::string>> mapStyles;
        std::map<std::string, int> blocCounts;
        std::set<std::string> blocKeys;
        std::map<std::string, std::shared_ptr<Bloc>> blocsSnapshot;
    }; // Struct pour sauvegarder l'état du document lors d'une transaction (pour les conditionnels)
    std::stack<TransactionState> transactionStack; 
};
