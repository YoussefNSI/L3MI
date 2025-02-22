#include "scanner.hh"
#include "driver.hh"

#include <iostream>
#include <fstream>
#include <variant>
#include <vector>
#include <map>
#include <string>
#include <memory>

#include <cstring>

std::shared_ptr<Document> doc = std::make_shared<Document>();

int main(int argc, char* argv[]) {
    Driver* driver = new Driver(doc);
    Scanner* scanner = new Scanner(std::cin, std::cout);
    yy::Parser* parser = new yy::Parser(*scanner, *driver);

    parser->parse();
    doc->HTMLtoFile("../sortie.html"); // Le fichier sera dans ProjetSRC

    delete driver;
    delete scanner;
    delete parser;

    return 0;
}
