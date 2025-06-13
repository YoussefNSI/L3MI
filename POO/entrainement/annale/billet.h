#pragma once

#include <string>
#include <memory>

class Billet{
public:
    Billet(std::string depart, std::string arrive, Billet::Type t, double p, int n) :
        _depart(depart), _arrivee(arrive), type(t), _prixBase(p), _nbVoyageurs(n){
        _numero = compteur++;
    }
    std::string getDepart() const { return _depart; }
    std::string getArrivee() const { return _arrivee; }

    // Ajouter ces déclarations dans la partie publique
    double prix_total() const;
    std::string tostring() const;


private:
    std::string _depart, _arrivee;
    int _numero;
    enum class Type { AllerSimple, AllerRetour } type;
    double _prixBase;
    int _nbVoyageurs;
    static int compteur;

};


class Client {
public:
    explicit Client(std::string nom) : _nom(nom) {}
    virtual ~Client() = default;
    std::string getNom() const { return _nom; }

private:
    std::string _nom;
};

class ClientReduction : public Client {
public:
    ClientReduction(std::string nom, int dateFinValidite)
        : Client(nom), _dateFinValidite(dateFinValidite) {}
    int getDateFinValidite() const { return _dateFinValidite; }

private:
    int _dateFinValidite;
};

class ClientRegulier : public ClientReduction {
public:
    static std::unique_ptr<ClientRegulier> fabrique_regulier(const std::string& nom,
                                           int dateFinValidite,
                                           const std::string& gare1,
                                           const std::string& gare2) {
        if (gare1 == gare2) return nullptr;
        return std::unique_ptr<ClientRegulier>(new ClientRegulier(nom, dateFinValidite, gare1, gare2));
    }

    std::string getGareDepart() const { return _gareDepart; }
    std::string getGareArrivee() const { return _gareArrivee; }

private:
    ClientRegulier(const std::string& nom, int dateFinValidite,
                  const std::string& gare1, const std::string& gare2)
        : ClientReduction(nom, dateFinValidite)
        , _gareDepart(gare1)
        , _gareArrivee(gare2) {}

    std::string _gareDepart;
    std::string _gareArrivee;
};

class ClientAccompagne : public ClientReduction {
public:
    static std::unique_ptr<ClientAccompagne> fabrique_accompagne(const std::string& nom,
                                               int dateFinValidite,
                                               int nbMaxVoyageurs) {
        if (nbMaxVoyageurs < 1 || nbMaxVoyageurs > 8) return nullptr;
        return std::unique_ptr<ClientAccompagne>(new ClientAccompagne(nom, dateFinValidite, nbMaxVoyageurs));
    }

    int getNbMaxVoyageurs() const { return _nbMaxVoyageurs; }

private:
    ClientAccompagne(const std::string& nom, int dateFinValidite, int nbMaxVoyageurs)
        : ClientReduction(nom, dateFinValidite)
        , _nbMaxVoyageurs(nbMaxVoyageurs) {}

    int _nbMaxVoyageurs;
};