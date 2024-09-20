#include <fstream>
#include <iostream>
#include <string>

std::string encodFile(std::string filename)
{
    std::ifstream fic(filename, std::ios::binary);
    if (!fic)
    {
        std::cerr << "Impossible d'ouvrir le fichier " << filename << std::endl;
        return "";
    }
    unsigned char c;
    std::string encodage;
    bool isAscii = true;
    bool isUtf8 = true;
    bool isLatin1 = true;

    while (fic >> c)
    {
        if (c > 127)
        {
            isAscii = false;
        }
        if ((c & 0xC0) == 0xC0)
        {
            int nbBits = 0;
            if ((c & 0xE0) == 0xC0)
            {
                nbBits = 1;
            }
            else if ((c & 0xF0) == 0xE0)
            {
                nbBits = 2;
            }
            else if ((c & 0xF8) == 0xF0)
            {
                nbBits = 3;
            }
            else
            {
                isUtf8 = false;
                break;
            }
            for (int i = 0; i < nbBits; i++)
            {
                if (!(fic >> c))
                {
                    isUtf8 = false;
                    break;
                }
                if ((c & 0xC0) != 0x80)
                {
                    isUtf8 = false;
                    break;
                }
            }
        }
        if (c > 255)
        {
            isLatin1 = false;
        }
    }

    if (isAscii)
    {
        encodage = "ASCII";
    }
    else if (isUtf8)
    {
        encodage = "UTF-8";
    }
    else if (isLatin1)
    {
        encodage = "ISO 8859-1";
    }
    return encodage;
}

int main()
{
    encodFile("text.ascii.txt");
    return 0;
}
