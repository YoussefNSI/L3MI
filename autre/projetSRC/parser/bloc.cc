#include <string>
#include <stdexcept>
#include <iostream>
#include "bloc.h"

std::string Titre::toHTML()
{
    std::string style;
    for (const auto &[key, value] : attributs)
    {
        style += key + ": " + value + "; ";
    }
    return "<h" + std::to_string(niveau) + " style=\"" + style + "\">" + texte + "</h" + std::to_string(niveau) + ">";
};

std::string Paragraphe::toHTML()
{
    std::string style;
    for (const auto &[key, value] : attributs)
    {
        style += key + ": " + value + "; ";
    }
    return "<p style=\"" + style + "\">" + texte + "</p>";
};

std::string Image::toHTML()
{
    return "<img src=\"" + src + "\" />";
}

std::string TitrePage::toHTML()
{
    return "<title>" + texte + "</title>";
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

const VariableType& Document::getVariable(const std::string &nom) const
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
    for (const auto &bloc : blocs)
    {
        html += bloc->toHTML();
    }
    return html;
}

std::string Commentaire::toHTML()
{
    return std::string();
}

void Document::afficherBlocs() const
{
    for (const auto &bloc : blocs)
    {
        std::cout << bloc->toHTML() << std::endl;
    }
}
