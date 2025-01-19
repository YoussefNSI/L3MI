#include "pacman.hh"
#include <algorithm>

/* classe position */

position::position(unsigned int x, unsigned int y) : _x(x), _y(y) {}
unsigned int position::x() const { return _x; }
unsigned int position::y() const { return _y; }
std::string position::to_string() const
{
    return "(" + std::to_string(_x) + "," + std::to_string(_y) + ")";
}
bool position::operator==(const position &p) const
{
    return _x == p._x && _y == p._y;
}
bool position::operator!=(const position &p) const
{
    return !(*this == p);
}
std::ostream& operator<<(std::ostream &os, const position &p)
{
    os << p.to_string();
    return os;
}

/* classe taille */

taille::taille(unsigned int largeur, unsigned int hauteur) : _largeur(largeur), _hauteur(hauteur) {}
unsigned int taille::w() const { return _largeur; }
unsigned int taille::h() const { return _hauteur; }
std::string taille::to_string() const
{
    return "(" + std::to_string(_largeur) + "," + std::to_string(_hauteur) + ")";
}
std::ostream &operator<<(std::ostream &os, const taille &t)
{
    os << t.to_string();
    return os;
}

/* classe element */

element::element(position const &pos, taille const & t) : _pos(pos), _t(t) {}
position element::pos() const { return _pos; }
taille element::tai() const { return _t; }
void element::setpos(const position &p)
{
    _pos = p;
}
char element::typeobjet() const { return 'X'; } // X = element non défini
std::ostream &operator<<(std::ostream &os, const element &e)
{
    os << "Position : " << e._pos.to_string() << " Taille : " << e._t.to_string();
    return os;
}
bool element::contient(const element &e) const
{
    return pos().x() <= e.pos().x() && pos().y() <= e.pos().y() &&
           (pos().x() + tai().w()) >= (e.pos().x() + e.tai().w()) &&
           (pos().y() + tai().h()) >= (e.pos().y() + e.tai().h());
}
bool element::intersection(const element &e) const
{
    return pos().x() < e.pos().x() + e.tai().w() && pos().x() + tai().w() > e.pos().x() &&
           pos().y() < e.pos().y() + e.tai().h() && pos().y() + tai().h() > e.pos().y();
}

/* classe pacman */

pacman::pacman(position pos, direction d) : element(pos, taille(13, 13)), _d(d), _invicibilite(0) {}
direction pacman::deplacement() const { return _d; }
void pacman::setdir(const direction &d) { _d = d; }
char pacman::typeobjet() const { return 'P'; }
bool pacman::invincible() const { return _invicibilite > 0; }
void pacman::decrementerinvincible()
{
    if (_invicibilite > 0)
        _invicibilite--;
}
void pacman::devenirinvincible() { _invicibilite = 200; }

/* classe fantome */

fantome::fantome(const position &pos,const direction &d) : element(pos, taille(20, 20)), _d(d) {}
direction fantome::deplacement() const { return _d; }
void fantome::setdir(const direction &d) { _d = d; }
char fantome::typeobjet() const { return 'F'; }

/* classe mur */

mur::mur(const position &pos, const taille &t) : element(pos, t)
{
    if (t.w() < 10 || t.h() < 10)
    {
        throw std::invalid_argument("La largeur et la hauteur d'un mur doivent être toutes les deux supérieures ou égales à 10.");
    }
}
char mur::typeobjet() const { return 'M'; }

/* classe pacgommes */

pacgommes::pacgommes(position const &pos) : element(pos, taille(3, 3)) {}
char pacgommes::typeobjet() const { return 'G'; }

/* classe exceptionjeu */

exceptionjeu::exceptionjeu(const std::string &message) : _message(message) {}
const char *exceptionjeu::what() const noexcept  { return _message.c_str(); }

/* classe jeu */

jeu::jeu() : _etat(etat::encours), _pacman(nullptr) {}
jeu::jeu(std::vector<std::shared_ptr<element>> elements) : _elements(elements), _etat(etat::encours), _pacman(nullptr) {}
jeu::jeu(const jeu &jeu) : _elements(jeu._elements), _etat(jeu._etat), _pacman(jeu._pacman) {}
jeu &jeu::operator=(const jeu &jeu)
{
    _elements = jeu._elements;
    _etat = jeu._etat;
    _pacman = jeu._pacman;
    return *this;
}
std::ostream &jeu::afficher(std::ostream &os) const
{
    for (auto e : _elements)
    {
        os << *e << std::endl;
    }
    switch (_etat)
    {
    case etat::encours:
        os << "Partie en cours" << std::endl;
        break;
    case etat::defaite:
        os << "Partie perdue" << std::endl;
        break;
    case etat::victoire:
        os << "Partie gagnée" << std::endl;
        break;
    }
    return os;
}
void jeu::ajouter(std::shared_ptr<element> e)
{
    if (std::any_of(_elements.begin(), _elements.end(), [e](std::shared_ptr<element> element)
                    { return element->intersection(*e); }))
    {
        throw exceptionjeu("L'élément à ajouter chevauche un autre élément.");
    }
    _elements.push_back(e);
}
void jeu::ajouterfantomes(int e)
{
    int x = 0;
    int y = 0;
    for (int i = 0; i < e; i++)
    {
        try
        {
            x = rand() % 320;
            y = rand() % 200;
            std::shared_ptr<fantome> f = std::make_shared<fantome>(position(x, y), direction::stop);
            ajouter(f);
        }
        catch (exceptionjeu &e)
        {
            i--;
        }
    }
}
void jeu::ajouterpacgommes(int e)
{
    int x = 0;
    int y = 0;
    for (int i = 0; i < e; i++)
    {
        try
        {
            x = rand() % 320;
            y = rand() % 200;
            std::shared_ptr<pacgommes> g = std::make_shared<pacgommes>(position(x, y));
            ajouter(g);
        }
        catch (exceptionjeu &e)
        {
            i--;
        }
    }
}
std::shared_ptr<pacman> jeu::accespacman()
{
    if (_pacman == nullptr)
    {
        auto it = std::find_if(_elements.begin(), _elements.end(), [](std::shared_ptr<element> e)
                               { return e->typeobjet() == 'P'; });
        if (it != _elements.end())
        {
            _pacman = std::dynamic_pointer_cast<pacman>(*it);
            return _pacman;
        }
        throw exceptionjeu("Le pacman n'a pas été trouvé.");
    }
    else
    {
        return _pacman;
    }
}
void jeu::directionjoueur(direction d)
{
    try
    {
        std::shared_ptr<pacman> p = accespacman();
        p->setdir(d);
    }
    catch (exceptionjeu &e)
    {
        std::cerr << e.what() << std::endl;
    }
}

void jeu::changerdirectionfantomes()
{
    std::vector<direction> dir = {direction::haut, direction::bas, direction::gauche, direction::droite, direction::stop};
    std::for_each(_elements.begin(), _elements.end(), [&dir](std::shared_ptr<element> e)
                  {
                      if (e->typeobjet() == 'F')
                      {
                          std::shared_ptr<fantome> f = std::dynamic_pointer_cast<fantome>(e);
                          int x = rand() % 5;
                          f->setdir(dir[x]);
                      }
                  });
}

void jeu::tourdejeu()
{
    appliquerdeplacementcollisionmur();
    appliquerdeplacementcontact();
    appliquerdeplacementmanger();
    changerdirectionfantomes();
    if (!std::any_of(_elements.begin(), _elements.end(), [](std::shared_ptr<element> element)
                     { return element->typeobjet() == 'G'; }))
    {
        _etat = etat::victoire;
    }
}

void jeu::appliquerdeplacementcollisionmur()
{
    for (auto e : _elements)
    {
        if (e->typeobjet() == 'P' || e->typeobjet() == 'F')
        {
            direction dir = direction::stop;
            int x = 0;
            int y = 0;
            if (e->typeobjet() == 'P')
            {
                dir = std::dynamic_pointer_cast<pacman>(e)->deplacement();
            }
            else
            {
                dir = std::dynamic_pointer_cast<fantome>(e)->deplacement();
            }
            switch (dir)
            {
            case direction::haut:
                y = -1;
                break;
            case direction::bas:
                y = 1;
                break;
            case direction::gauche:
                x = -1;
                break;
            case direction::droite:
                x = 1;
                break;
            case direction::stop:
                break;
            }
            if (x != 0 || y != 0)
            {
                bool collision = false;
                std::shared_ptr<element> e2 = std::make_shared<element>(position(e->pos().x() + x, e->pos().y() + y), e->tai());
                if (std::any_of(_elements.begin(), _elements.end(), [e2](std::shared_ptr<element> element)
                                { return (element->typeobjet() == 'M' && element->intersection(*e2)); }))
                {
                    collision = true;
                }
                if (!collision)
                {
                    e->setpos(position(e->pos().x() + x, e->pos().y() + y));
                }
            }
        }
    }
}

void jeu::appliquerdeplacementcontact()
{
    std::shared_ptr<pacman> p = accespacman();
    if (std::any_of(_elements.begin(), _elements.end(), [p](std::shared_ptr<element> element)
                    { return (element->typeobjet() == 'F' && element->intersection(*p)); }))
    {
        if (p->invincible())
        {
            _elements.erase(std::remove_if(_elements.begin(), _elements.end(), [p](std::shared_ptr<element> element)
                                           { return (element->typeobjet() == 'F' && element->intersection(*p)); }),
                            _elements.end());
        }
        else
        {
            _etat = etat::defaite;
        }
    }
}

void jeu::appliquerdeplacementmanger()
{
    std::shared_ptr<pacman> p = accespacman();
    if (std::any_of(_elements.begin(), _elements.end(), [p](std::shared_ptr<element> element)
                    { return (element->typeobjet() == 'G' && element->intersection(*p)); }))
    {
        _elements.erase(std::remove_if(_elements.begin(), _elements.end(), [p](std::shared_ptr<element> element)
                                       { return (element->typeobjet() == 'G' && element->intersection(*p)); }),
                        _elements.end());
        p->devenirinvincible();
    }
}
