#include <iostream>
#include "sequence.hh"


int main()
{
    sequence a;
    a.ajouter(couleur::rouge);
    a.ajouter(couleur::bleu);
    a.ajouter(couleur::rouge);
    a.ajouter(couleur::vert);
    sequence b(a);
    a.afficher(a.acces(0)); std::cout<< "\n";
    for(indicesequence i=0; i<a.taille(); ++i){
        a.afficher(a.acces(i)); std::cout<< " ";
    }
    std::cout<< "\n";
    a.vider();
    std::cout << "A: "; a.afficher(std::cout); std::cout<< "\n";
    std::cout << "B: "; b.afficher(std::cout); std::cout<< "\n";
    a.copier(b);
    std::cout << a.comparer(b) << "\n";
    return 0;
}
