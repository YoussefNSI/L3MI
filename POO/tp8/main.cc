#include "sequence.hh"
#include "simon.hh"
#include <qapplication.h>



int main(int argc, char *argv[]) {
    QTranslator qtTranslator;
    if(!qtTranslator.load("qt_" + QLocale::system().name(), QLibraryInfo::path(QLibraryInfo::TranslationsPath)))
        return 1;
    QApplication app(argc, argv);
    app.installTranslator(&qtTranslator);
    sequence s;
    s.ajouter(couleur::rouge);
    s.ajouter(couleur::bleu);
    s.ajouter(couleur::jaune);
    s.ajouter(couleur::vert);
    simon f(s);
    f.show();
    return app.exec();
}
