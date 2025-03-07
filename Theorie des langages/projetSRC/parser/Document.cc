#include "Document.h"
#include <string>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <regex>

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

std::shared_ptr<Bloc> Document::getNBloc(const std::string &type, int index) const
{
    int count = 0;
    for (const auto &bloc : blocs)
    {
        if ((type == "p" && std::dynamic_pointer_cast<Paragraphe>(bloc.second)) ||
            (type == "t" && std::dynamic_pointer_cast<Titre>(bloc.second)) ||
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

void Document::beginTransaction() {
    std::cout << "Begin transaction" << std::endl;
    TransactionState state;
    state.variables = variables;
    state.proprietes = proprietes;
    state.mapStyles = mapStyles;
    state.blocCounts = blocCounts;
    
    // On sauvegarde les blocs
    for (const auto& pair : blocs) {
        state.blocKeys.insert(pair.first);
        state.blocsSnapshot[pair.first] = pair.second->clone(); // On clone pour les éventuelles modifications sur le bloc
    }
    
    transactionStack.push(state);
}

void Document::commitTransaction() {
    // On vide la pile
    if (!transactionStack.empty()) {
        transactionStack.pop();
        std::cout<< "Commit" << std::endl;
    }
}

void Document::rollbackTransaction() {
    if (transactionStack.empty()) return;

    TransactionState state = transactionStack.top();
    transactionStack.pop();

    std::cout << "Rollback" << std::endl;

    // On restaure les états sauvegardés
    variables = state.variables;
    proprietes = state.proprietes;
    mapStyles = state.mapStyles;
    blocCounts = state.blocCounts;

    // On supprime les blocs qui ont été ajoutés depuis le début de la transaction
    std::vector<std::string> toRemove;
    for (const auto& pair : blocs) {
        if (state.blocKeys.find(pair.first) == state.blocKeys.end()) {
            std::cout << "Remove " << pair.first << std::endl;
            toRemove.push_back(pair.first);
        }
    }
    for (const auto& key : toRemove) {
        std::cout << "Remove (2)" << key << std::endl;
        blocs.erase(key);
    }

    // On restaure les blocs qui ont été modifiés
    for (const auto& [key, snapshotBloc] : state.blocsSnapshot) {
        if (blocs.find(key) != blocs.end()) {
            std::cout << "Restore " << key << std::endl;
            blocs[key] = snapshotBloc->clone(); // Rétablir l'état original
        }
    }
}

std::string Document::toHTML() const
{
    // On met à jour les propriétés de chaque bloc pour voir si il y a eu des changements depuis leur ajout
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
