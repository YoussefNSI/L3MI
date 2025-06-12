#pragma once

#include <string>

class Billet{
public:
    Billet(std::string depart, std::string arrive, Billet::Type t, double p, int n) :
        _depart(depart), _arrivee(arrive), type(t), _prixBase(p), _nbVoyageurs(n){
        _numero = compteur++;
    }
    std::string getDepart() const { return _depart; }
    std::string getArrivee() const { return _arrivee; }


private:
    std::string _depart, _arrivee;
    int _numero;
    enum class Type { AllerSimple, AllerRetour } type;
    double _prixBase;
    int _nbVoyageurs;
    static int compteur;

};
