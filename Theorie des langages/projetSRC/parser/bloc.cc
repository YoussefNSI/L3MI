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
    // Gestion des propriétés
    for (const auto &[propriete, mapStyle] : mapStyles)
    {
        bool tout, paragraphe = false;
        std::pair<bool, int> titre = std::make_pair(false, 0);
        std::regex titre_pattern("^titre([1-9])$");
        std::smatch matches;

        if (propriete == "page")
        {
            tout = true;
        }
        else if (propriete == "paragraphe")
        {
            paragraphe = true;
        }

        if (std::regex_match(propriete, matches, titre_pattern))
        {
            titre.first = true;
            titre.second = std::stoi(matches[1].str());
            std::cout << "Niveau extrait : " << titre.second << std::endl;
        }
        else
        {
            std::cerr << "Format de titre invalide" << std::endl;
        }

        for (const auto &bloc : blocs)
        {
            if (tout && (bloc.second->getType() == "Titre" || bloc.second->getType() == "Paragraphe"))
            {
                for (const auto &[prop, valeur] : mapStyle)
                {
                    bloc.second->setPropriete(prop, valeur);
                }
            }
            else if (paragraphe && bloc.second->getType() == "Paragraphe")
            {
                for (const auto &[prop, valeur] : mapStyle)
                {
                    bloc.second->setPropriete(prop, valeur);
                }
            }
            else if (titre.first && bloc.second->getType() == "Titre" && std::dynamic_pointer_cast<Titre>(bloc.second)->getNiveau() >= titre.second)
            {
                for (const auto &[prop, valeur] : mapStyle)
                {
                    bloc.second->setPropriete(prop, valeur);
                }
            }
        }
    }

    // On commence à écrire le HTML maintenant que tout est à jour

    std::string html;
    int indentLevel = 0;
    auto indent = [&]()
    { return std::string(indentLevel * 4, ' '); };

    // En-tête HTML
    html += "<!DOCTYPE html>\n";
    html += indent() + "<html lang=\"fr\">\n";
    indentLevel++;

    // Section <head>
    html += indent() + "<head>\n";
    indentLevel++;

    // TitrePage dans <title>
    if (metablocs.second != nullptr && metablocs.first == "TitrePage")
    {
        html += indent() + metablocs.second->toHTML(0) + "\n";
    }

    // Métadonnées
    std::string encodage = indent() + "<meta charset=\"utf-8\">";
    std::string autresMeta;

    for (const auto &[prop, valeur] : proprietes)
    {
        if (prop == "encodage")
        {
            encodage = indent() + "<meta charset=\"" + valeur + "\">\n";
        }
        else if (prop == "icone")
        {
            autresMeta += indent() + "<link rel=\"icon\" href=\"" + valeur + "\">\n";
        }
        else if (prop == "css")
        {
            autresMeta += indent() + "<link rel=\"stylesheet\" href=\"" + valeur + "\">\n";
        }
        else if (prop == "langue")
        {
            size_t pos = html.find("lang=\"fr\"");
            if (pos != std::string::npos)
            {
                html.replace(pos + 6, 2, valeur);
            }
        }
    }

    html += encodage;
    html += autresMeta;
    indentLevel--;
    html += "\n" + indent() + "</head>\n";

    // Section <body>
    html += indent() + "<body>\n";
    indentLevel++;
    std::string commentaire = "";
    std::string body;

    for (const auto &bloc : blocs)
    {
        if (bloc.second->getType() == "Commentaire")
        {
            if (commentaire == "")
            {
                commentaire = indent() + "<!--\n";
            }
            commentaire += bloc.second->toHTML(indentLevel);
        }
        else
        {
            body += bloc.second->toHTML(indentLevel) + "\n";
        }
    }
    if (commentaire != "")
    {
        html += commentaire + indent() + "-->\n";
    }
    html += body;
    indentLevel--;
    html += indent() + "</body>\n";

    indentLevel--;
    html += indent() + "</html>";

    return html;
}

void Document::HTMLtoFile(const std::string &filename) const
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << toHTML();
        file.close();
    }
    else
    {
        throw std::runtime_error("Impossible d'ouvrir le fichier " + filename);
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

Document::Document()
{
    blocCounts["paragraphe"] = 0;
    blocCounts["titre"] = 0;
    blocCounts["image"] = 0;
    blocCounts["titrepage"] = 0;
    blocCounts["commentaire"] = 0;
}

void Document::afficherBlocs() const
{
    for (const auto &pair : blocs)
    {
        std::cout << pair.first << " -> " << pair.second->getType() << std::endl;
    }
}

std::shared_ptr<Bloc> Document::getNBloc(const std::string &type, int index) const
{
    int count = 0;
    for (const auto &bloc : blocs)
    {
        std::cout << "Bloc de type " << bloc.first << " Valeur : " << bloc.second->getTexte() << std::endl;
        if ((type == "p" && std::dynamic_pointer_cast<Paragraphe>(bloc.second)) ||
            (type == "h" && std::dynamic_pointer_cast<Titre>(bloc.second)) ||
            (type == "img" && std::dynamic_pointer_cast<Image>(bloc.second)))
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

void Document::addBloc(std::shared_ptr<Bloc> bloc)
{
    std::string type = bloc->getType();
    ++blocCounts[type];
    if (type == "TitrePage" && blocCounts[type] > 1)
    {
        throw std::runtime_error("Un seul bloc de type TitrePage est autorisé");
    }
    else if (type == "TitrePage")
    {
        blocCounts[type] = 1;
        metablocs = std::make_pair(type, bloc);
    }
    else
    {
        std::string key = type + std::to_string(blocCounts[type]);
        blocs[key] = bloc;
    }
}