#include "contexte.hh"
#include "expressionBinaire.hh"
#include "constante.hh"
#include "variable.hh"

#include <iostream>

int main( int  argc, char* argv[]) {
    Contexte c1, c2;

    c1["x"] = 1; // x = 1  dans le Contexte C1
    c2["x"] = 5; // x = 5  dans le Contexte C2

    ExpressionPtr const2 = std::make_shared<Constante>(2); // constante 2
    ExpressionPtr x = std::make_shared<Variable>("x"); // variable "x"
    ExpressionPtr xpc2 = std::make_shared<ExpressionBinaire>(const2,x, OperateurBinaire::plus); // 2 + x
    ExpressionPtr exp = std::make_shared<ExpressionBinaire>(xpc2, const2, OperateurBinaire::multiplie); // (2 + x) * 2

    std::cout << "Contexte 1 : " << exp->calculer(c1) << std::endl;
    std::cout << "Contexte 2 : " << exp->calculer(c2) << std::endl;

    return 0;
}
