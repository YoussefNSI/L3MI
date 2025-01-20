#include "sequence.hh"
#include "simon.hh"
#include <qapplication.h>



int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    sequence s;
    s.ajouter(couleur::rouge);
    s.ajouter(couleur::bleu);
    s.ajouter(couleur::jaune);
    s.ajouter(couleur::vert);
    simon f(s);
    f.show();
    return app.exec();
}
