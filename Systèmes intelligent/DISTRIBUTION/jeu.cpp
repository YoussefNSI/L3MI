#include "../DISTRIBUTION/jeu.h"
#include <iostream>
#define PARTIE_NON_TERMINEE 2
#define PARTIE_NULLE 0
#define ALIGNEMENT 1

#include "plateau.h"

int _tab[MAX_LARGEUR];
int _nombre;
int _puissance[MAX_LARGEUR];
int _alignement;
int _dual_x[MAX_HAUTEUR];
int _dual_o[MAX_HAUTEUR];
int _nb_tours;


Plateau _plateau =  Plateau();

Jeu::Jeu() {
  reset();
}

void Jeu::reset() {
  int i = 0;
  for (i = 0; i < MAX_HAUTEUR; i++) {
    _dual_o[i] = 0;
    _dual_x[i] = 0;
  }
  for (i = 0; i < MAX_LARGEUR; i++) {
    _puissance[i] = 1;
    _plateau._hauteur[i] = 0;
    _plateau._pions[i] = 0;
    _tab[i] = i; 
  }
  _alignement = false;
  _nombre = MAX_LARGEUR;
  _etat._val = PARTIE_NON_TERMINEE;
  _nb_tours = 0;
}




bool Jeu::coup_licite(int coup_relatif) {
  return((coup_relatif >= 1) && (coup_relatif <= _nombre));
}

  
void Jeu::joue(int indice_coup) {
  _nb_tours++;
  if (_nb_tours%2) {indice_coup = -indice_coup;}

  int indice_relatif =  abs(indice_coup)-1;
  int indice_absolu = _tab[indice_relatif];

  int hauteur = _plateau._hauteur[indice_absolu];
  
  if (hauteur == MAX_HAUTEUR-1) {
    _nombre--;
    int i;
    for (i = indice_relatif ; i < _nombre ; i++) {
      _tab[i] = _tab[i+1];
    }
  }
  

  if (indice_coup > 0) {
    _plateau._pions[indice_absolu] += _puissance[indice_absolu];

            _dual_x[hauteur] = _dual_x[hauteur] + (1 << indice_absolu);
  }
  else {
    _dual_o[hauteur] = _dual_o[hauteur] + (1 << indice_absolu);
  }
  _puissance[indice_absolu] = _puissance[indice_absolu] << 1;
  _plateau._hauteur[indice_absolu]++;


  int le_coup = _plateau._pions[indice_absolu];
  if (indice_coup > 0) {
    if ((hauteur == 3 && le_coup == 15) || // xxxx
  	(hauteur == 4 && hauteur > 0 && le_coup == 30) || // oxxxx
  	(hauteur == 5 && ((le_coup == 60) || // ooxxxx
  			       (le_coup == 61))) // xoxxxx
  	) { _etat._val = ALIGNEMENT; return;}
  }
  else {
    if ((hauteur == 3  && le_coup == 0) || // oooo
  	(hauteur == 4  && le_coup == 1) || // xoooo
  	(hauteur == 5  && (le_coup == 2 || // oxoooo
  			   le_coup == 3))    // xxoooo
  	) { _etat._val = ALIGNEMENT; return;}
  }
  // horizontal
  int dual;
  if (indice_coup > 0) 
    dual = _dual_x[hauteur];
  else 
    dual = _dual_o[hauteur];
  
  if ((indice_absolu < 4 && (dual & 1) && (dual & 2) && (dual & 4) && (dual & 8)) || // ????___
  	((indice_absolu > 0 && indice_absolu < 5) && (dual & 2) && (dual & 4) && (dual & 8) && (dual & 16)) || // _????__
  	((indice_absolu > 1 && indice_absolu < 6) && (dual & 4) && (dual & 8) && (dual & 16) && (dual & 32)) || // __????_
  	(indice_absolu > 2 && (dual & 8) && (dual & 16) && (dual & 32) && (dual & 64)) // ___????
      )
    { _etat._val = ALIGNEMENT; return;}

  if (_nb_tours == MAX_HAUTEUR * MAX_LARGEUR) {_etat._val = PARTIE_NULLE;}
  
}


int Jeu::nb_coups() const {
  return _nombre;
}

bool Jeu::terminal() const{
  return (_etat._val != PARTIE_NON_TERMINEE);
}

bool Jeu::victoire() {
  return ((_etat._val != PARTIE_NON_TERMINEE) && (_etat._val == ALIGNEMENT) && (_nb_tours%2)); 
}


bool Jeu::pat() {
  return ((_etat._val != PARTIE_NON_TERMINEE) && (_etat._val == PARTIE_NULLE));
}
