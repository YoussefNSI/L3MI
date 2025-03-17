#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <ctime> // ajouté

#define RACINE 0

class Etat
{
public:
  int _val; // valeur de la racine
};

class Jeu
{
private:
  Etat _etat;

  unsigned int etats[17];

public:
  Jeu();
  void reset();
  bool coup_licite(int); // Vérifie si le coup est licite
  void joue(int);        // Joue le coup (sans vérifier qu'il est licite)
  int nb_coups() const;  // Le nombre de coups possibles
  bool terminal() const; // Vérifie si un état final a été atteint (dans ce cas le jeu prend fin)
  bool pat();
  bool victoire();

  // ajouté :

  int get_etat() const { return _etat._val; }
  int random(int n) // entre 1 et n inclus
  {
    std::srand(std::time({}));
    int lowest = 1;
    int range = (n - lowest) + 1;

    return lowest + rand() % range;
  }
};
