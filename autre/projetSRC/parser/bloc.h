#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <variant>

class Bloc
{
public:
    virtual ~Bloc() = default;
    virtual std::string toHTML() = 0;
};

class Titre : public Bloc
{
public:
    Titre(std::map<std::string, std::string> attributs, std::string texte, int niveau)
        : attributs(attributs), texte(texte), niveau(niveau) {}

    int getNiveau() const { return niveau; }
    std::string toHTML() override;
    std::string getTexte() const { return texte; }
    std::map<std::string, std::string> getAttributs() const { return attributs; }

private:
    std::map<std::string, std::string> attributs;
    std::string texte;
    int niveau;
};

class Paragraphe : public Bloc
{
public:
    Paragraphe(std::map<std::string, std::string> attributs, std::string texte)
        : attributs(attributs), texte(texte) {}

    std::string toHTML() override;
    std::string getTexte() const { return texte; }
    std::map<std::string, std::string> getAttributs() const { return attributs; }

private:
    std::map<std::string, std::string> attributs;
    std::string texte;
};

class Image : public Bloc
{
public:
    Image(std::string src) : src(src) {}
    std::string toHTML() override;
    std::string getSrc() const { return src; }

private:
    std::string src;
};

class TitrePage : public Bloc
{
public:
    TitrePage(std::string texte) : texte(texte) {}
    std::string toHTML() override;
    std::string getTexte() const { return texte; }

private:
    std::string texte;
};

class Commentaire : public Bloc
{
public:
    Commentaire(std::string texte) : texte(texte) {}
    std::string toHTML() override;
    std::string getTexte() const { return texte; }

private:
    std::string texte;
};

using VariableType = std::variant<int, std::string, Bloc *>;

class Document
{
public:
    Document() = default;
    ~Document()
    {
        for (auto bloc : blocs)
        {
            delete bloc;
        }
    }
    void addBloc(Bloc *bloc)
    {
        blocs.push_back(bloc);
    }
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
    void afficherBlocs() const;
    std::string getPropriete(const std::string &nom) const;
    const VariableType &getVariable(const std::string &nom) const;
    std::map<std::string, std::string> getStyle(const std::string &nom) const;
    std::string toHTML() const;

private:
    std::vector<Bloc *> blocs;
    std::map<std::string, std::string> proprietes;                       // @DEFINE
    std::map<std::string, VariableType> variables;                       // variables
    std::map<std::string, std::map<std::string, std::string>> mapStyles; // attributs
};