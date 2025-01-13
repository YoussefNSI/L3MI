#include "pacman.hh"

void jeu::ajouter(element *e)
{
    if (std::any_of(_elements.begin(), _elements.end(), [e](element *element)
                    { return element->intersection(*e); }))
    {
        throw exceptionjeu("L'élément à ajouter chevauche un autre élément.");
    }
    _elements.push_back(e);
}

void jeu::ajouterfantome(int e)
{
    int x = 0;
    int y = 0;
    for (int i = 0; i < e; i++)
    {
        try
        {
            x = rand() % 320;
            y = rand() % 200;
            fantome *f = new fantome(position(x, y), direction::stop);
            ajouter(f);
        }
        catch (exceptionjeu &e)
        {
            i--;
        }
    }
}

pacman *jeu::accespacman()
{
    if(_pacman == nullptr)
    {
        auto it = std::find_if(_elements.begin(), _elements.end(), [](element *e)
                               { return e->typeobjet() == 'P'; });
        if (it != _elements.end())
        {
            _pacman = dynamic_cast<pacman*>(*it);
            return dynamic_cast<pacman*>(*it);
        }
        throw exceptionjeu("Le pacman n'a pas été trouvé.");
    }
    else
    {
        return _pacman;
    }
}

void jeu::directionjoueur(direction d){
    try {
        pacman *p = accespacman();
        p->setdir(d);
    } catch (exceptionjeu &e) {
        std::cerr << e.what() << std::endl;
    }
}

void jeu::changerdirectionfantomes(){
    std::vector<direction> dir = {direction::haut, direction::bas, direction::gauche, direction::droite, direction::stop};
    std::for_each(_elements.begin(), _elements.end(), [&dir](element* e){
        if(e->typeobjet() == 'F'){
            fantome *f = dynamic_cast<fantome*>(e);
            int x = rand() % 5;
            f->setdir(dir[x]);
        }
    });
}

void jeu::tourdejeu(){
    appliquerdeplacementcollisionmur();
    appliquerdeplacementcontact();
    appliquerdeplacementmanger();
    changerdirectionfantomes();
    if(!std::any_of(_elements.begin(), _elements.end(), [](element *element)
                    { return element->typeobjet() == 'G'; })){
        _etat = etat::victoire;
    }
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

void jeu::appliquerdeplacementcollisionmur()
{
    for (auto e : _elements)
    {
        if(e->typeobjet() == 'P' || e->typeobjet() == 'F'){
            direction dir = direction::stop;
            int x = 0;
            int y = 0;
            if(e->typeobjet() == 'P'){
                dir = dynamic_cast<pacman*>(e)->deplacement();
            }
            else{
                dir = dynamic_cast<fantome*>(e)->deplacement();
            }
            switch(dir){
                case direction::haut:
                    y = -1;
                    break;
                case direction::bas:
                    y = 1;
                    break;
                case direction::gauche:
                    x = -1;;
                    break;
                case direction::droite:
                    x = 1;;
                    break;
                case direction::stop:
                    break;
            }
            if(x != 0 || y != 0){
                bool collision = false;
                element *e2 = new element(position(e->pos().x() + x, e->pos().y() + y), e->tai());
                if(std::any_of(_elements.begin(), _elements.end(), [e2](element *element)
                    { return (element->typeobjet() == 'M' && element->intersection(*e2)); })){
                    collision = true;
                }
                delete e2;
                if(!collision){
                    e->setpos(position(e->pos().x() + x, e->pos().y() + y));
                }
            }
        }
    }
}

void jeu::appliquerdeplacementcontact(){
    pacman *p = accespacman();
    if(std::any_of(_elements.begin(), _elements.end(), [p](element *element)
                    { return (element->typeobjet() == 'F' && element->intersection(*p)); })){
        if(p->invincible()){
            _elements.erase(std::remove_if(_elements.begin(), _elements.end(), [p](element *element)
                    { return (element->typeobjet() == 'F' && element->intersection(*p)); }), _elements.end());
        }
        else{
            _etat = etat::defaite;
        }
    }
}

void jeu::appliquerdeplacementmanger(){
    pacman *p = accespacman();
    if(std::any_of(_elements.begin(), _elements.end(), [p](element *element)
                    { return (element->typeobjet() == 'G' && element->intersection(*p)); })){
        _elements.erase(std::remove_if(_elements.begin(), _elements.end(), [p](element *element)
                    { return (element->typeobjet() == 'G' && element->intersection(*p)); }), _elements.end());
        p->devenirinvincible();
    }
}

