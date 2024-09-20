#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>

std::string convertEncodage(const std::string &texte, const std::string &fromEncodage, const std::string &toEncodage)
{
    std::string resultat;

    if (fromEncodage == "Latin1" && toEncodage == "UTF-8")
    {
        for (unsigned char c : texte)
        {
            if (c < 0x80)
            {
                resultat += c;
            }
            else
            {
                resultat += 0xC0 | (c >> 6);
                resultat += 0x80 | (c & 0x3F);
            }
        }
    }
    else if (fromEncodage == "UTF-8" && toEncodage == "Latin1")
    {
        for (size_t i = 0; i < texte.size(); ++i)
        {
            unsigned char c = texte[i];
            if (c < 0x80)
            {
                resultat += c;
            }
            else if ((c & 0xE0) == 0xC0)
            {
                unsigned char c1 = texte[i++];
                unsigned char c2 = texte[i];
                resultat += ((c1 & 0x1F) << 6) | (c2 & 0x3F);
            }
            else
            {
                throw std::runtime_error("Caractère non représentable en Latin-1");
            }
        }
    }
    else if (fromEncodage == "UTF-8" && toEncodage == "ASCII")
    {
        for (size_t i = 0; i < texte.size(); ++i)
        {
            unsigned char c = texte[i];
            if (c < 0x80)
            {
                resultat += c;
            }
            else
            {
                throw std::runtime_error("Caractère non représentable en ASCII");
            }
        }
    }
    else if (fromEncodage == "ASCII" && toEncodage == "UTF-8")
    {
        for (unsigned char c : texte)
        {
            if (c < 0x80)
            {
                resultat += c;
            }
            else
            {
                throw std::runtime_error("Caractère non représentable en UTF-8");
            }
        }
    }
    else
    {
        throw std::runtime_error("Encodage non supporté : " + fromEncodage + " -> " + toEncodage);
    }

    return resultat;
}

std::string lireFic(const std::string &fichier)
{
    std::ifstream fic(fichier, std::ios::binary);
    if (!fic)
    {
        throw std::runtime_error("Erreur lors de l'ouverture du fichier: " + fichier);
    }
    std::string contenu((std::istreambuf_iterator<char>(fic)), std::istreambuf_iterator<char>());
    return contenu;
}

std::string convertEncodageFile(const std::string &fichier, const std::string &fromEncodage, const std::string &toEncodage)
{
    std::string texte = lireFic(fichier);
    return convertEncodage(texte, fromEncodage, toEncodage);
}

int main()
{

    convertEncodageFile("text.utf8.txt", "UTF-8", "Latin1");
    return 0;
}
