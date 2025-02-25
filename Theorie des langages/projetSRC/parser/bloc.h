#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <variant>
#include <set>
#include <stack>

class Bloc
{
public:
    virtual ~Bloc() = default;
    virtual std::string toHTML(int currentIndent=0) = 0;
    virtual void setPropriete(const std::string &nom, const std::string &valeur) = 0;
    virtual std::string getPropriete(const std::string &nom) { return ""; };
    virtual std::string getType() const = 0;
    virtual std::string getTexte() const = 0;
    virtual std::shared_ptr<Bloc> clone() const = 0;
};

class Titre : public Bloc
{
public:
    Titre(std::map<std::string, std::string> attributs, std::string texte, int niveau)
        : attributs(attributs), texte(texte), niveau(niveau) {}

    int getNiveau() const { return niveau; }
    std::string toHTML(int currentIndent) override;
    std::string getTexte() const override { return texte; }
    std::map<std::string, std::string> getAttributs() const { return attributs; }

    void setPropriete(const std::string &nom, const std::string &valeur) override;
    std::string getPropriete(const std::string &nom) override;
    std::string getType() const override { return "Titre"; }
    std::shared_ptr<Bloc> clone() const override {
        return std::make_shared<Titre>(*this);
    }

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

    std::string toHTML(int currentIndent) override;
    std::string getTexte() const override { return texte; }
    std::map<std::string, std::string> getAttributs() const { return attributs; }

    void setPropriete(const std::string &nom, const std::string &valeur) override;
    std::string getPropriete(const std::string &nom) override;
    std::string getType() const override { return "Paragraphe"; }
    std::shared_ptr<Bloc> clone() const override {
        return std::make_shared<Paragraphe>(*this);
    }

private:
    std::map<std::string, std::string> attributs;
    std::string texte;
};

class Image : public Bloc
{
public:
    Image(std::string src) : src(src) {}
    std::string toHTML(int currentIndent) override;
    std::string getTexte() const override { return src; }
    std::string getSrc() const { return src; }

    void setPropriete(const std::string &nom, const std::string &valeur) override;
    std::string getType() const override { return "Image"; }
    std::shared_ptr<Bloc> clone() const override {
        return std::make_shared<Image>(*this);
    }

private:
    std::string src;
};

class TitrePage : public Bloc
{
public:
    TitrePage(std::string texte) : texte(texte) {}
    std::string toHTML(int currentIndent) override;
    std::string getTexte() const override { return texte; }

    void setPropriete(const std::string &nom, const std::string &valeur) override;
    std::string getType() const override { return "TitrePage"; }
    std::shared_ptr<Bloc> clone() const override {
        return std::make_shared<TitrePage>(*this);
    }

private:
    std::string texte;
};

class Commentaire : public Bloc
{
public:
    Commentaire(std::string texte) : texte(texte) {}
    std::string toHTML(int currentIndent) override;
    std::string getTexte() const override { return texte; }

    void setPropriete(const std::string &nom, const std::string &valeur) override;
    std::string getType() const override { return "Commentaire"; }
    std::shared_ptr<Bloc> clone() const override {
        return std::make_shared<Commentaire>(*this);
    }

private:
    std::string texte;
};
