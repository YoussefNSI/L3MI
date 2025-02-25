#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <regex>
#include "bloc.h"

std::string Titre::toHTML(int currentIndent)
{
    std::string espace(currentIndent * 4, ' ');
    std::string style, html;
    if (attributs.size() > 0)
    {
        for (const auto &[key, value] : attributs)
        {
            style += key + ": " + value + "; ";
        }
    }
    if (style.empty())
    {
        html = espace + "<h" + std::to_string(niveau) + (style.empty() ? ">" : " style=\"" + style + "\">") + texte + "</h" + std::to_string(niveau) + ">";
    }
    else
    {
        html = espace + "<h" + std::to_string(niveau) + " style=\"" + style + "\">" + texte + "</h" + std::to_string(niveau) + ">";
    }
    return html;
};

void Titre::setPropriete(const std::string &nom, const std::string &valeur)
{
    attributs[nom] = valeur;
}

std::string Titre::getPropriete(const std::string &nom)
{
    auto it = attributs.find(nom);
    if (it != attributs.end())
    {
        return it->second;
    }
    throw std::runtime_error("Propriété " + nom + " introuvable");
}

std::string Paragraphe::toHTML(int currentIndent)
{
    std::string espace(currentIndent * 4, ' ');
    std::string style;
    if (attributs.size() > 0)
    {
        for (const auto &[key, value] : attributs)
        {
            style += key + ": " + value + "; ";
        }
    }

    std::istringstream iss(texte);
    std::string line;
    std::string html;
    int i = 1;
    html = espace + "<p" + (style.empty() ? ">" : " style=\"" + style + "\">");

    while (std::getline(iss, line))
    {
        if (i >= 2)
            html += "\n" + espace;
        html += line;
        i++;
    }

    html += "</p>";
    return html;
}

void Paragraphe::setPropriete(const std::string &nom, const std::string &valeur)
{
    attributs[nom] = valeur;
}

std::string Paragraphe::getPropriete(const std::string &nom)
{
    auto it = attributs.find(nom);
    if (it != attributs.end())
    {
        return it->second;
    }
    throw std::runtime_error("Propriété " + nom + " introuvable");
}

std::string Image::toHTML(int currentIndent)
{
    std::string espace(currentIndent * 4, ' ');
    return espace + "<img src=\"" + src + "\" />";
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

std::string TitrePage::toHTML(int currentIndent)
{
    std::string espace(currentIndent * 4, ' ');
    return espace + "<title>" + texte + "</title>";
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

std::string Commentaire::toHTML(int currentIndent)
{
    std::string espace(currentIndent * 4, ' ');
    std::istringstream iss(texte);
    std::string line;
    std::string html;
    std::string indent = espace;

    while (std::getline(iss, line))
    {
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos)
        {
            continue;
        }
        size_t end = line.find_last_not_of(" \t");
        std::string trimmedLine = line.substr(start, end - start + 1);
        html += indent + trimmedLine + "\n";
    }
    return html;
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