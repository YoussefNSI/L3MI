#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define MAX_HAUTEUR 6
#define MAX_LARGEUR 7

class Plateau {
 public :
  int _hauteur[MAX_LARGEUR];
  int _pions[MAX_LARGEUR];

  void afficher( std::ostream &flux) const;
};
