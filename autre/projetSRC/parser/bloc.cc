#include <string>
#include <stdexcept>
#include <iostream>
#include "bloc.h"

int Document::nbParagraphe = 0;
int Document::nbTitre = 0;
int Document::nbImage = 0;
int Document::nbTitrePage = 0;
int Document::nbCommentaire = 0;

std::string Titre::toHTML()
{
    std::string style;
    for (const auto &[key, value] : attributs)
    {
        style += key + ": " + value + "; ";
    }
    return "<h" + std::to_string(niveau) + " style=\"" + style + "\">" + texte + "</h" + std::to_string(niveau) + ">";
};

void Titre::setPropriete(const std::string &nom, const std::string &valeur)
{
    attributs[nom] = valeur;
}

std::string Paragraphe::toHTML()
{
    std::string style;
    for (const auto &[key, value] : attributs)
    {
        style += key + ": " + value + "; ";
    }
    return "<p style=\"" + style + "\">" + texte + "</p>";
};

void Paragraphe::setPropriete(const std::string &nom, const std::string &valeur)
{
    attributs[nom] = valeur;
}

std::string Image::toHTML()
{
    return "<img src=\"" + src + "\" />";
}

void Image::setPropriete(const std::string &nom, const std::string &valeur)
{
    if (nom == "src")
    {
        src = valeur;
    }
    else
    {
        throw std::runtime_error("Propriété " + nom + " non supportée pour Image");
    }
}

std::string TitrePage::toHTML()
{
    return "<title>" + texte + "</title>";
}

void TitrePage::setPropriete(const std::string &nom, const std::string &valeur)
{
    if (nom == "texte")
    {
        texte = valeur;
    }
    else
    {
        throw std::runtime_error("Propriété " + nom + " non supportée pour TitrePage");
    }
}

std::string Document::getPropriete(const std::string &nom) const
{
    auto it = proprietes.find(nom);
    if (it != proprietes.end())
    {
        return it->second;
    }
    throw std::runtime_error("Propriété " + nom + " introuvable");
}

const VariableType &Document::getVariable(const std::string &nom) const
{
    auto it = variables.find(nom);
    if (it != variables.end())
    {
        return it->second;
    }
    throw std::runtime_error("Variable " + nom + " introuvable");
}

std::map<std::string, std::string> Document::getStyle(const std::string &nom) const
{
    auto it = mapStyles.find(nom);
    if (it != mapStyles.end())
    {
        return it->second;
    }
    throw std::runtime_error("Style " + nom + " introuvable");
}

std::string Document::toHTML() const
{
    std::string html;

    for (const auto &[prop, valeur] : proprietes)
    {
        if (prop == "encodage")
        {
            html += "<!DOCTYPE html>\n<html lang=\"fr\">\n<head>\n<meta charset=\"" + valeur + "\">\n";
        }
        else if (prop == "icone")
        {
            html += "<link rel=\"icon\" href=\"" + valeur + "\">\n";
        }
        else if (prop == "css")
        {
            html += "<link rel=\"stylesheet\" href=\"" + valeur + "\">\n";
        }
        else if (prop == "langue")
        {
            html += "<html lang=\"" + valeur + "\">\n";
        }
    }

    return html;
}

std::string Commentaire::toHTML()
{
    return std::string();
}

void Commentaire::setPropriete(const std::string &nom, const std::string &valeur)
{
    if (nom == "texte")
    {
        texte = valeur;
    }
    else
    {
        throw std::runtime_error("Propriété " + nom + " non supportée pour Commentaire");
    }
}

void Document::afficherBlocs() const
{
    for (const auto &bloc : blocs)
    {
        std::cout << bloc.second->toHTML() << std::endl;
    }
}

Bloc *Document::getNBloc(const std::string &type, int index) const
{
    int count = 0;
    for (const auto &bloc : blocs)
    {
        if ((type == "p" && dynamic_cast<Paragraphe *>(bloc.second)) ||
            (type == "h" && dynamic_cast<Titre *>(bloc.second)) ||
            (type == "img" && dynamic_cast<Image *>(bloc.second)))
        {
            if (count == index)
            {
                return bloc.second;
            }
            count++;
        }
    }
    throw std::runtime_error("Bloc de type " + type + " avec l'indice " + std::to_string(index) + " introuvable");
}

void Document::addBloc(const std::string &type, Bloc *bloc)
{
    switch(type[0])
    {
        case 'p':
            nbParagraphe++;
            break;
        case 't':
            nbTitre++;
            break;
        case 'i':
            nbImage++;
            break;
        case 'z':
            nbTitrePage++;
            break;
        case 'c':
            nbCommentaire++;
            break;
        default:
            throw std::runtime_error("Type " + type + " inconnu");
    }
    blocs[type] = bloc;
}